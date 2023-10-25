#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}

import skimage
from skimage import measure, morphology

def extract_connnectivity(binary_img):
    connet_area = measure.label(binary_img, connectivity=8)
    cmax = connet_area.max()
    for i in range(cmax, 1):
        connect
    return connet_area

def keep_largest_connected_component(nda):
    nda_output = nda
    label = measure.label((nda>0))#.astype(np.uint8))
    maxArea = 0
    for region in measure.regionprops(label):
        if region.area > maxArea:
            maxArea = region.area
    mask = morphology.remove_small_objects(label, maxArea -1)
    nda_output[mask==0] = 0
    return nda_output


if __name__ == "__main__":
    import nibabel as nib
    import os
    print("===")

    ########
    #test_dir = "/media/yaols/PortSSD001/Original data/CRC_110/重勾画107/CRC ROI"
    test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/4. full_attn_coord"
    #test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/3. nnunet_coord"
    #test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/2. nnunet_full_attn"
    #test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/1. nnunet"
    ref_dir = "/media/yaols/PortSSD001/MICCAI results/results107/0. labels"
    test_files = os.listdir(test_dir)
    ref_files = os.listdir(ref_dir)
    print(test_dir)
    num = len(ref_files)
    HD_all = []
    MSD_all = []
    DC_all = []

    for i in range(num):
        print(i, ref_files[i])
        test = nib.load(os.path.join(test_dir, ref_files[i])).get_fdata()
        header = nib.load(os.path.join(test_dir, ref_files[i])).header
        affine = nib.load(os.path.join(test_dir, ref_files[i])).affine
        ref = nib.load(os.path.join(ref_dir, ref_files[i])).get_fdata()
        test[np.where(test==1)]=0
        test[np.where(test==2)]=1
        ref[np.where(ref==1)]=0
        ref[np.where(ref==2)]=1
        test= keep_largest_connected_component(test)
        print(test.max(),ref.max())
        DC = dice(test,ref)
        HD_95 = hausdorff_distance_95(test, ref)
        MSD = avg_surface_distance(test, ref)
        print(DC,HD_95,MSD)
        if DC<0.1: #np.isnan(HD_95):
            print("=============large penalty============")
            # HD_all.append(300)
            # MSD_all.append(300)
        else:
            HD_all.append(HD_95)
            MSD_all.append(MSD)
        DC_all.append(DC)
        label = nib.Nifti1Image(test, header = header, affine = affine)
        compose_path = os.path.join('/media/yaols/PortSSD001/MICCAI results/results107/our',ref_files[i])
        nib.save(label, compose_path)
    print(HD_all)
    print(MSD_all)
    print(DC_all)
    print("===================================================")
    print("the values of nnunet is")
    print("HD_95 is:",np.mean(HD_all),np.std(HD_all), "MSD is:",np.mean(MSD_all),np.std(MSD_all))
    print(np.mean(DC_all), np.std(DC_all))




# ###########
#     test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/2. nnunet_full_attn"
#     ref_dir = "/media/yaols/PortSSD001/MICCAI results/results107/0. labels"
#     test_files = os.listdir(test_dir)
#     ref_files = os.listdir(ref_dir)
#     print(test_dir)
#     num = len(test_files)
#     HD_all = []
#     MSD_all = []

#     for i in range(num):
#         print(i, ref_files[i])
#         test = nib.load(os.path.join(test_dir, ref_files[i])).get_fdata()
#         ref = nib.load(os.path.join(ref_dir, ref_files[i])).get_fdata()
#         test[np.where(test==1)]=0
#         test[np.where(test==2)]=1
#         ref[np.where(ref==1)]=0
#         ref[np.where(ref==2)]=1
#         HD_95 = hausdorff_distance_95(test, ref)
#         MSD = avg_surface_distance_symmetric(test, ref)
#         print(HD_95,MSD)
#         if np.isnan(HD_95):
#             print("=============large penalty============")
#             HD_all.append(1000)
#             MSD_all.append(200)
#         else:
#             HD_all.append(HD_95)
#             MSD_all.append(MSD)

#     print("===================================================")
#     print("the values of nnunet is")
#     print("HD_95 is:",np.mean(HD_all),np.std(HD_all), "MSD is:",np.mean(MSD_all),np.std(MSD_all))


# # ############
#     test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/3. nnunet_coord"
#     ref_dir = "/media/yaols/PortSSD001/MICCAI results/results107/0. labels"
#     test_files = os.listdir(test_dir)
#     ref_files = os.listdir(ref_dir)
#     print(test_dir)
#     num = len(test_files)
#     HD_all = []
#     MSD_all = []

#     for i in range(num):
#         print(i, ref_files[i])
#         test = nib.load(os.path.join(test_dir, ref_files[i])).get_fdata()
#         ref = nib.load(os.path.join(ref_dir, ref_files[i])).get_fdata()
#         test[np.where(test==1)]=0
#         test[np.where(test==2)]=1
#         ref[np.where(ref==1)]=0
#         ref[np.where(ref==2)]=1
#         HD_95 = hausdorff_distance_95(test, ref)
#         MSD = avg_surface_distance_symmetric(test, ref)
#         print(HD_95,MSD)
#         if np.isnan(HD_95):
#             print("=============large penalty============")
#             HD_all.append(1000)
#             MSD_all.append(200)
#         else:
#             HD_all.append(HD_95)
#             MSD_all.append(MSD)

#     print("===================================================")
#     print("the values of nnunet is")
#     print("HD_95 is:",np.mean(HD_all),np.std(HD_all), "MSD is:",np.mean(MSD_all),np.std(MSD_all))


# # ############
#     test_dir = "/media/yaols/PortSSD001/MICCAI results/results107/4. full_attn_coord"
#     ref_dir = "/media/yaols/PortSSD001/MICCAI results/results107/0. labels"
#     test_files = os.listdir(test_dir)
#     ref_files = os.listdir(ref_dir)
#     print(test_dir)
#     num = len(test_files)
#     HD_all = []
#     MSD_all = []

#     for i in range(num):
#         print(i, ref_files[i])
#         test = nib.load(os.path.join(test_dir, ref_files[i])).get_fdata()
#         ref = nib.load(os.path.join(ref_dir, ref_files[i])).get_fdata()
#         test[np.where(test==1)]=0
#         test[np.where(test==2)]=1
#         ref[np.where(ref==1)]=0
#         ref[np.where(ref==2)]=1
#         HD_95 = hausdorff_distance_95(test, ref)
#         MSD = avg_surface_distance_symmetric(test, ref)
#         print(HD_95,MSD)
#         if np.isnan(HD_95):
#             print("=============large penalty============")
#             HD_all.append(1000)
#             MSD_all.append(200)
#         else:
#             HD_all.append(HD_95)
#             MSD_all.append(MSD)

#     print("===================================================")
#     print("the values of nnunet is")
#     print("HD_95 is:",np.mean(HD_all),np.std(HD_all), "MSD is:",np.mean(MSD_all),np.std(MSD_all))


#  ###########
#     test_dir = "/media/yaols/PortSSD001/Original data/CRC_110/重勾画107/CRC ROI"
#     ref_dir = "/media/yaols/PortSSD001/MICCAI results/results107/0. labels"
#     test_files = os.listdir(test_dir)
#     ref_files = os.listdir(ref_dir)
#     print(test_dir)
#     num = len(test_files)
#     HD_all = []
#     MSD_all = []

#     for i in range(num):
#         print(i, ref_files[i])
#         test = nib.load(os.path.join(test_dir, ref_files[i])).get_fdata()
#         ref = nib.load(os.path.join(ref_dir, ref_files[i])).get_fdata()
#         test[np.where(test==1)]=0
#         test[np.where(test==2)]=1
#         ref[np.where(ref==1)]=0
#         ref[np.where(ref==2)]=1
#         HD_95 = hausdorff_distance_95(test, ref)
#         MSD = avg_surface_distance_symmetric(test, ref)
#         print(HD_95,MSD)
#         if np.isnan(HD_95):
#             print("=============large penalty============")
#             HD_all.append(1000)
#             MSD_all.append(200)
#         else:
#             HD_all.append(HD_95)
#             MSD_all.append(MSD)

#     print("===================================================")
#     print("the values of nnunet is")
#     print("HD_95 is:",np.mean(HD_all),np.std(HD_all), "MSD is:",np.mean(MSD_all),np.std(MSD_all))
