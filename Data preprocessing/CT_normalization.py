import argparse
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def CT_resampling(ct_root, ct_data, resampled_root):
    for data in ct_data:
        if 'roi' not in data:
            C, nii, gz = data.split('.')
            try:
                c, num = C.split('-')
            except:
                num = C
            num = str(int(num))

            img_root = os.path.join(ct_root, data)
            img = sitk.ReadImage(img_root)
            img_np = sitk.GetArrayFromImage(img)
            img_min = int(img_np.min())
            size = img.GetSize()

            reference_np = np.zeros((size[2], size[1], size[0]))
            reference_img = sitk.GetImageFromArray(reference_np)
            reference_img.CopyInformation(img)
            reference_img.SetSpacing([1.0, 1.0, 1.0])

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_img)
            resampler.SetInterpolator(sitk.sitkBSplineResamplerOrder3)
            resampler.SetDefaultPixelValue(img_min)

            orig_size = np.array(img.GetSize(), dtype=np.int)
            orig_spacing = img.GetSpacing()
            new_size = orig_size*(orig_spacing)
            new_size = np.ceil(new_size).astype(np.int)
            new_size = [int(s) for s in new_size]
            resampler.SetSize(new_size)

            new_img = resampler.Execute(img)
            save_pth = os.path.join(resampled_root, num)
            if not os.path.isdir(save_pth):
                os.makedirs(save_pth)

            resample_path = os.path.join(save_pth, data)
            sitk.WriteImage(new_img, resample_path)

def ROI_resampling(ct_root, ct_data, resampled_root):
    for data in ct_data:
        if 'roi' in data:
            C, roi, nii, gz = data.split('.')
            try:
                c, num = C.split('-')
            except:
                num = C
            num = str(int(num))

            roi_root = os.path.join(ct_root, data)
            roi = sitk.ReadImage(roi_root)
            roi_np = sitk.GetArrayFromImage(roi)
            roi_min = int(roi_np.min())
            size = roi.GetSize()

            reference_np = np.zeros((size[2], size[1], size[0]))
            reference_img = sitk.GetImageFromArray(reference_np)
            reference_img.CopyInformation(roi)
            reference_img.SetSpacing([1.0, 1.0, 1.0])

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(roi_min)

            orig_size = np.array(roi.GetSize(), dtype=np.int)
            orig_spacing = roi.GetSpacing()
            new_size = orig_size*(orig_spacing)
            new_size = np.ceil(new_size).astype(np.int)
            new_size = [int(s) for s in new_size]
            resampler.SetSize(new_size)

            new_roi = resampler.Execute(roi)

            save_pth = os.path.join(resampled_root, num)
            if not os.path.isdir(save_pth):
                os.makedirs(save_pth)

            resample_roi_path = os.path.join(save_pth, data)
            sitk.WriteImage(new_roi, resample_roi_path)

def image_slice_check(ct_root, ct_data):
    thick = {}
    z_thick = {}
    for data in ct_data:
        if 'roi' in data :
            C, roi, nii, gz = data.split('.')
            try:
                c, num = C.split('-')
            except:
                num = C
            num = str(int(num))

            roi_origin_root = os.path.join(ct_root, data)
            roi_origin = sitk.ReadImage(roi_origin_root)
            spacing = roi_origin.GetSpacing()
            thick[num] = spacing
            z_thick[num] = roi_origin.GetSpacing()[2]

    thickness = [value for key, value in z_thick.items()]
    thickness_set = set(thickness)
    thick_count = {}
    for i in thickness_set:
        thick_count[i] = 0
        for j in thickness :
            if i == j :
                thick_count[i] += 1
    return thick_count

def ROI_max_slice(resampled_root):
    per_patient_max_slice = np.array([])
    per_patient_max_slice_dict = {}
    per_patient_total_max_slice_dict = {}
    patient_no = os.listdir(resampled_root)
    for no in patient_no:
        patient_root = os.path.join(resampled_root, no)
        patient_data = os.listdir(patient_root)
        for data in patient_data:
            if 'roi' in data:
                roi_root = os.path.join(patient_root, data)
                roi = sitk.ReadImage(roi_root)
                roi_np = sitk.GetArrayFromImage(roi)
                roi_min = int(roi_np.min())

                per_slice_roi_pixel_count = np.array([])
                for i in range(len(roi_np)):
                    roi_z = roi_np[i]
                    roi_count = len(np.where(roi_z != roi_min)[0])
                    per_slice_roi_pixel_count = np.append(per_slice_roi_pixel_count, roi_count)

                max_slice = np.max(per_slice_roi_pixel_count)
                z_slice_no = np.where(per_slice_roi_pixel_count == max_slice)[0]
                per_patient_max_slice = np.append(per_patient_max_slice, z_slice_no)
                per_patient_total_max_slice_dict[str(no)] = z_slice_no
                per_patient_max_slice_dict[str(no)] = z_slice_no[0]


    return per_patient_max_slice, per_patient_max_slice_dict, per_patient_total_max_slice_dict

def ROI_column_raw_check(resampled_root, per_patient_max_slice_dict):
    row_col_check_dict = {}
    row_col_check_np = np.array([])
    row_col_check_np = row_col_check_np.reshape(0, 4)
    patient_no = os.listdir(resampled_root)
    for no in patient_no:
        patient_root = os.path.join(resampled_root, no)
        patient_data = os.listdir(patient_root)
        col_check = np.array([])
        row_check = np.array([])

        for data in patient_data:
            if 'roi' in data:
                roi_root = os.path.join(patient_root, data)
                roi_img = sitk.ReadImage(roi_root)
                roi_np = sitk.GetArrayFromImage(roi_img)
                roi_min = int(roi_np.min())

                max_slice_no = per_patient_max_slice_dict[str(no)]
                max_slice = roi_np[max_slice_no]
                for i in range(max_slice.shape[0]):
                    roi_count = np.where(max_slice[i] != roi_min)[0]
                    col_check = np.append(col_check, roi_count)
                col_check_unique = np.unique(col_check)

                for j in range(max_slice.shape[1]):
                    roi_count = np.where(max_slice[:, j] != roi_min)[0]
                    row_check = np.append(row_check, roi_count)
                row_check_unique = np.unique(row_check)

                row_min = int(row_check_unique.min())
                row_max = int(row_check_unique.max())
                col_min = int(col_check_unique.min())
                col_max = int(col_check_unique.max())

                idx = [row_min, row_max, col_min, col_max]
                idx_np = np.array(idx)
                idx_np = idx_np.reshape(1, 4)

        row_col_check_dict[no] = idx
        row_col_check_np = np.append(row_col_check_np, idx_np, axis = 0)
    return row_col_check_dict, row_col_check_np

def ROI_center(row_col_check_dict):
    roi_size_per_patient_dict = {}
    roi_size_dict = {}
    roi_size_per_patient_np = np.array([])
    roi_size_per_patient_np = roi_size_per_patient_np.reshape(0, 2)
    roi_center_per_patient_dict = {}
    roi_center_per_patient_np = np.array([])
    roi_center_per_patient_np = roi_center_per_patient_np.reshape(0, 2)

    for key in row_col_check_dict:
        length = row_col_check_dict[key][1] - row_col_check_dict[key][0]
        mid_length = int((row_col_check_dict[key][1] + row_col_check_dict[key][0]) / 2)
        width = row_col_check_dict[key][3] - row_col_check_dict[key][2]
        mid_width = int((row_col_check_dict[key][3] + row_col_check_dict[key][2]) / 2)
        size = [length, width]
        center = [mid_length, mid_width]
        size_np = np.array(size)
        size_np = size_np.reshape(1, 2)
        center_np = np.array(center)
        center_np = center_np.reshape(1, 2)
        roi_size_per_patient_dict[key] = size
        roi_size_dict[key] = max(size)
        roi_size_per_patient_np = np.append(roi_size_per_patient_np, size_np, axis = 0)
        roi_center_per_patient_dict[key] = center
        roi_center_per_patient_np = np.append(roi_center_per_patient_np, center_np, axis = 0)
    return roi_center_per_patient_dict, roi_size_dict

def axial_slice_three(resampled_root, np_root, per_patient_max_slice_dict, roi_center_per_patient_dict, cut_size = 64):
    patient_no = os.listdir(resampled_root)

    for no in patient_no :
        patient_root = os.path.join(resampled_root, no)
        patient_data = os.listdir(patient_root)
        patient_np_root = os.path.join(np_root, no)

        if not os.path.isdir(patient_np_root):
            os.makedirs(patient_np_root)

        for data in patient_data:
            if 'roi' in data:
                roi_root = os.path.join(patient_root, data)
                roi_img = sitk.ReadImage(roi_root)
                roi_np = sitk.GetArrayFromImage(roi_img)
                roi_min = roi_np.min()

                roi_background = np.full((3, 128, 128), roi_min)
                max_slice_no = per_patient_max_slice_dict[str(no)]

                max_slice_roi_pre = roi_np[max_slice_no - 3]
                max_slice_roi_mid = roi_np[max_slice_no]
                max_slice_roi_post = roi_np[max_slice_no + 3]

                roi_row_center = roi_center_per_patient_dict[no][0]
                roi_col_center = roi_center_per_patient_dict[no][1]

                row_start_point = roi_row_center - cut_size
                row_end_point = roi_row_center + cut_size
                col_start_point = roi_col_center - cut_size
                col_end_point = roi_col_center + cut_size

                roi_cut_pre = max_slice_roi_pre[row_start_point:row_end_point, col_start_point:col_end_point]
                roi_cut_mid = max_slice_roi_mid[row_start_point:row_end_point, col_start_point:col_end_point]
                roi_cut_post = max_slice_roi_post[row_start_point:row_end_point, col_start_point:col_end_point]

                if max_slice_roi_mid.min() == -1024:
                    roi_cut_pre = roi_cut_pre + 1024
                    roi_cut_mid = roi_cut_mid + 1024
                    roi_cut_post = roi_cut_post + 1024

                for i in range(roi_cut_pre.shape[0]):
                    for j in range(roi_cut_pre.shape[1]):
                        roi_background[0][i][j] = roi_cut_pre[i][j]
                        roi_background[1][i][j] = roi_cut_mid[i][j]
                        roi_background[2][i][j] = roi_cut_post[i][j]

                np.save(patient_np_root + '/roi.npy', roi_background)

            else:
                ct_root = os.path.join(patient_root, data)
                ct_img = sitk.ReadImage(ct_root)
                ct_np = sitk.GetArrayFromImage(ct_img)
                ct_min = ct_np.min()

                ct_background = np.full((3, 128, 128), ct_min)
                max_slice_no = per_patient_max_slice_dict[str(no)]

                max_slice_ct_pre = ct_np[max_slice_no - 3]
                max_slice_ct_mid = ct_np[max_slice_no]
                max_slice_ct_post = ct_np[max_slice_no + 3]

                roi_row_center = roi_center_per_patient_dict[no][0]
                roi_col_center = roi_center_per_patient_dict[no][1]

                row_start_point = roi_row_center - cut_size
                row_end_point = roi_row_center + cut_size
                col_start_point = roi_col_center - cut_size
                col_end_point = roi_col_center + cut_size
                if row_start_point < 0:
                    row_start_point = 0
                if row_end_point < 0 :
                    row_end_point = 0
                if col_start_point < 0:
                    col_start_point = 0
                if col_end_point < 0:
                    col_end_point = 0

                ct_cut_pre = max_slice_ct_pre[row_start_point:row_end_point, col_start_point:col_end_point]
                ct_cut_mid = max_slice_ct_mid[row_start_point:row_end_point, col_start_point:col_end_point]
                ct_cut_post = max_slice_ct_post[row_start_point:row_end_point, col_start_point:col_end_point]

                for i in range(ct_cut_pre.shape[0]) :
                    for j in range(ct_cut_pre.shape[1]):
                        ct_background[0][i][j] = ct_cut_pre[i][j]
                        ct_background[1][i][j] = ct_cut_mid[i][j]
                        ct_background[2][i][j] = ct_cut_post[i][j]

                np.save(patient_np_root + '/ct.npy', ct_background)

def min_intensity(np_root):
    no = os.listdir(np_root)
    min_check = {}
    max_check = {}
    range_check = {}
    for num in no:
        num_root = os.path.join(np_root, num)
        data = os.listdir(num_root)
        for nps in data:
            if 'ct' in nps:
                ct_root = os.path.join(num_root, nps)
                ct = np.load(ct_root)
                min_value = ct.min()
                min_check[num] = min_value
                max_value = ct.max()
                max_check[num] = max_value
                range_check[num] = max_value - min_value

    modified_patient = [key for key, values in min_check.items() if int(values) < -1024 ]

    for m_no in modified_patient:
        m_root = os.path.join(np_root, m_no)
        data = os.listdir(num_root)

        for ct_np in data:
            if 'ct' in ct_np:
                ct_root = os.path.join(m_root, ct_np)
                ct = np.load(ct_root)

                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        for k in range(ct.shape[2]):
                            if int(ct[i][j][k]) < -1024:
                                ct[i][j][k] = -1024

                np.save(m_root + '/ct_m.npy', ct)

def normalization(np_root):
    np_number = os.listdir(np_root)
    for number in np_number:
        patient_np_root = os.path.join(np_root, number)
        data = os.listdir(patient_np_root)
        for nps in data:
            if nps == 'ct_m.npy':
                ct_np_root = os.path.join(patient_np_root, nps)
                ct_np = np.load(ct_np_root)
                ct_normalization = (ct_np - (-1024)) / (3071 - (-1024))
                np.save(patient_np_root + '/ct_normalized.npy', ct_normalization)
            elif nps == 'ct.npy':
                ct_np_root = os.path.join(patient_np_root, nps)
                ct_np = np.load(ct_np_root)
                ct_normalization = (ct_np - (-1024)) / (3071 - (-1024))
                np.save(patient_np_root + '/ct_normalized.npy', ct_normalization)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_root",
        default='/DATA/data_cancer/',
        type=str,
        required=True,
    )


    args = parser.parse_args()

    ct_root = os.path.join(args.data_root, 'image')
    np_root = os.path.join(args.data_root, 'numpy')
    resampled_root = os.path.join(args.data_root, 'resampled')

    if not os.path.isdir(np_root):
        os.makedirs(np_root)
    if not os.path.isdir(resampled_root):
        os.makedirs(resampled_root)

    ct_data = os.listdir(ct_root)

    patient_number = []
    for i in ct_data:
        if 'roi' not in i:
            C, nii, gz = i.split('.')
            try:
                c, num = C.split('-')
            except:
                num = C
            patient_number.append(str(int(num)))

    print('CT resampling is proceeding....')
    CT_resampling(ct_root, ct_data, resampled_root)
    print('ROI resampling is proceeding.....')
    ROI_resampling(ct_root, ct_data, resampled_root)

    thick_count = image_slice_check(ct_root, ct_data)
    per_patient_max_slice, per_patient_max_slice_dict, per_patient_total_max_slice_dict = ROI_max_slice(resampled_root)
    row_col_check_dict, row_col_check_np = ROI_column_raw_check(resampled_root, per_patient_max_slice_dict)
    roi_center_per_patient_dict, roi_size_dict = ROI_center(row_col_check_dict)

    roi_max_size = max([value for (key, value) in roi_size_dict.items()])
    cut_size = max(64, roi_max_size / 2)

    print('extracting 3 axial slices.....')
    axial_slice_three(resampled_root, np_root, per_patient_max_slice_dict, roi_center_per_patient_dict, cut_size = 64)
    min_intensity(np_root)
    print('Finally normalization is proceeding.....')
    normalization(np_root)


if __name__ == "__main__":
    main()
