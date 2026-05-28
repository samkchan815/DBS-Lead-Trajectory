import SimpleITK as sitk
import nibabel as nib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ants

import os
import glob
from config import ATLAS_DIR, DATA_DIR, OUTPUT_DIR

import SimpleITK as sitk

def check_mi(img1, img2):
    '''Check if mutula information can be calculated.'''
    try:
        mi = ants.image_mutual_information(img1, img2)
        return mi
    except RuntimeError as e:
        if "do not sufficiently overlap" in str(e):
            print(f"Warning: Images don't overlap sufficiently. Cannot compute mutual information.")
            return None
        else:
            raise e

def ants_registration(mni, anat, output_file, atlas, atlas_output):
    '''Image registration utilizing ANTs package. Returns original and registered mutual information values.'''
    fixed = ants.image_read(mni) # read in mni image
    moving = ants.image_read(anat) # read in patient image
    atlas_labels = ants.image_read(atlas) # read in atlas image

    # change orientation of atlas if needed to match mni space
    if not ants.get_orientation(fixed) == ants.get_orientation(atlas_labels):
        fixed = ants.reorient_image2(fixed, ants.get_orientation(atlas_labels))

    # fix orientations
    if not ants.get_orientation(moving) == ants.get_orientation(atlas_labels):
        print(f"Reorienting to match atlas orientation...")
        moving = ants.reorient_image2(moving, ants.get_orientation(atlas_labels))
        # print all orientations
        print(ants.get_orientation(moving), ants.get_orientation(fixed), ants.get_orientation(atlas_labels))


    # register image using SyN transformation (non-linear)
    mytx = ants.registration(fixed=fixed, 
                            moving=moving, 
                            type_of_transform='SyN')
    
    # save the registered image
    ants.image_write(mytx['warpedmovout'], output_file)
    print("Registration complete. Output saved to {output_file}\n")

    miOg = check_mi(fixed, moving) # original MI
    if miOg is not None:
        print(f"Original Mutual Information: {miOg:.4f}")
    mi = check_mi(fixed, mytx['warpedmovout']) # registered MI
    print(f"Mutual Information: {mi:.4f}")

    # apply transformations to brain atlas
    transformed_atlas = ants.apply_transforms(
        fixed=moving,
        moving=atlas_labels,
        transformlist=mytx['invtransforms'],
        interpolator='nearestNeighbor'
    )
    
    # write transformed atlas to file
    ants.image_write(transformed_atlas, atlas_output)
    print(f"Atlas in patient space saved to {atlas_output}\n")

    return miOg, mi

def get_patient_id(path):
    '''Extract patient ID from file path. (Assumes patient ID starts with "PDa")'''
    parts = path.split(os.sep)
    for part in parts:
        if part.startswith("PDa"): 
            return part
    return None

def check_ct_seg_mapping(ct_files, seg_masks):
    '''Check that each CT file has a corresponding segmentation mask'''
    # create dictionaries mapping patient ID to file paths
    ct_map = {get_patient_id(f): f for f in ct_files if get_patient_id(f) is not None}
    seg_map = {get_patient_id(f): f for f in seg_masks if get_patient_id(f) is not None}

    # find matching patient IDs
    matching_ids = set(ct_map.keys()).intersection(seg_map.keys())


    print("Patients with both CT and segmentation:")
    for pid in sorted(matching_ids):
        print(pid)

    # match corresponding file paths
    matching_ct_files = [ct_map[pid] for pid in matching_ids]
    matching_seg_files = [seg_map[pid] for pid in matching_ids]

    print(f"\nNumber of matching CT files: {len(matching_ct_files)}")
    print(f"Number of matching segmentation files: {len(matching_seg_files)}")

    ct_only_ids = set(ct_map.keys()) - set(seg_map.keys())
    print("CT files without a matching segmentation:")
    for pid in sorted(ct_only_ids):
        print(pid)

    # check for ct files without segmentation
    ct_only_files = [ct_map[pid] for pid in ct_only_ids]
    print(f"\nNumber of CTs without segmentation: {len(ct_only_files)}")

    if ct_only_files != 0:
        return 1
    else:
        return 0
    
def visualize_registration(fixed, moving, registered, atlas, final_atlas, patient_id, output_dir='results'):
    '''Get overlay of patient image on MNI and transformed atlas on patient image.'''
    fixed_data = fixed.numpy() # pull image numpy arrays
    moving_data = moving.numpy()
    registered_data = registered.numpy()
    atlas_data = atlas.numpy()
    final_atlas_data = final_atlas.numpy()

    slice_index = fixed_data.shape[2] // 2  # fixed middle slice
    moving_index = moving_data.shape[2] // 2 # CT image middle slice

    plt.figure(figsize=(10, 8)) # create figure

    # Registered Image Overlay on MNI
    plt.subplot(1, 2, 1)
    plt.title('Registered Image Overlay on MNI')
    plt.imshow(fixed_data[:, :, slice_index], cmap='gray')
    plt.imshow(registered_data[:, :, slice_index], alpha=0.7, cmap='gray')
    plt.axis('off')

    # Transformed Atlas Overlay on original CT
    plt.subplot(1, 2, 2)
    plt.title('Transformed Atlas')
    plt.imshow(moving_data[:, :, moving_index], cmap='gray')
    plt.imshow(final_atlas_data[:, :, moving_index], alpha=0.7)
    plt.axis('off')

    output_filename = os.path.join(output_dir, f'{patient_id}_registration_overlay.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Overlay visualization saved to: {output_filename}")

def visualize_all_steps(fixed, moving, atlas, registered, final_atlas):
    '''Visualize MNI, CT, Registered CT, Atlas, and Transformed Atlas.'''
    # pull image numpy arrays
    fixed_data = fixed.numpy() 
    moving_data = moving.numpy()
    registered_data = registered.numpy()
    atlas_data = atlas.numpy()
    final_atlas_data = final_atlas.numpy()

    slice_index = fixed_data.shape[2] // 2  # Fixed middle slice
    moving_index = moving_data.shape[2] // 2 # CT image middle slice

    plt.figure(figsize=(10, 8)) # create figure

    # MNI Image
    plt.subplot(2, 3, 1)
    plt.title('MNI Image')
    plt.imshow(fixed_data[:, :, slice_index], cmap='gray')
    plt.axis('off')

    # CT Image
    plt.subplot(2, 3, 2)
    plt.title('Moving Image')
    plt.imshow(moving_data[:, :, moving_index], cmap='gray')
    plt.axis('off')

    # Registered Image to MNI Space
    plt.subplot(2, 3, 3)
    plt.title('Registered Image')
    plt.imshow(registered_data[:, :, slice_index], cmap='gray')
    plt.axis('off')

    # Brain Atlas
    plt.subplot(2, 3, 4)
    plt.title('Brain Atlas')
    plt.imshow(atlas_data[:, :, slice_index])
    plt.axis('off')

    # Registered Image Overlay on MNI
    plt.subplot(2, 3, 5)
    plt.title('Registered Image Overlay on MNI')
    plt.imshow(fixed_data[:, :, slice_index], cmap='gray')
    plt.imshow(registered_data[:, :, slice_index], alpha=0.7, cmap='gray')
    plt.axis('off')

    # Transformed Atlas Overlay on original CT
    plt.subplot(2, 3, 6)
    plt.title('Transformed Atlas')
    plt.imshow(moving_data[:, :, moving_index], cmap='gray')
    plt.imshow(final_atlas_data[:, :, moving_index], alpha=0.7)
    plt.axis('off')

    plt.show() # display figure

def parse_atlas_labels(label_file):
    '''Parse the atlas label file to create a mapping of label numbers to brain region names.'''
    label_dict = {} # initialize dictionary
    line_num= 1 # label numbers start at 1
    delimiter = '('

    # allow passing either a basename or full path
    if not os.path.isabs(label_file):
        label_file = os.path.join(ATLAS_DIR, label_file)

    with open(label_file, 'r') as f: # open file as read only
        for line in f: # loop through each line in label text file
            line=line.replace(')', '') # remove closing parenthesis
            parts = line.strip().split(delimiter) # split by delimiter

            # save to dictionary in mini list
            # 0: full label name 1: abbreviation
            label_dict[line_num] = [parts[0].strip(), parts[1].strip()] # add to dict
            line_num += 1 # track label number (line # = label #s)
    return label_dict

def create_vector(segmentation_file, output_atlas_file, atlas_label_file):
    '''Create a vector representation of atlas regions present in the segmentation.'''
    seg_img = ants.image_read(segmentation_file)
    atlas_img = ants.image_read(output_atlas_file)

    # fix orientation of segmentation to match atlas if needed
    if not ants.get_orientation(seg_img) == ants.get_orientation(atlas_img):
        print(f"Reorienting segmentation to match atlas orientation...")
        seg_img = ants.reorient_image2(seg_img, ants.get_orientation(atlas_img))
        print(ants.get_orientation(seg_img), ants.get_orientation(atlas_img))

    seg_arr = seg_img.numpy() # convert to image arrays
    atlas_arr = atlas_img.numpy()

    multiplied = seg_arr * atlas_arr
    unique, counts = np.unique(multiplied[multiplied>0], return_counts=True)

    with open(atlas_label_file) as f:
        atlas_regions = [line.strip() for line in f.readlines()]
    n_regions = len(atlas_regions)+1  # get number of regions
    print(f"Number of regions: {n_regions}")

    bin_vector = np.zeros(n_regions) # initialize vector

    for i in unique: # enter binary into vector
        if i < n_regions: 
            bin_vector[int(i)] = 1

    return bin_vector
    
def multiple_img_registration(input_dir, output_dir, mni_file, atlas_file, output_xlsx_path, atlas_labels_file):
    base_dir = input_dir

    mi_df = pd.DataFrame(columns=['Patient_ID', 'Original_MI', 'Registered_MI']) # initialize MI DataFrame

    if not os.path.exists(output_dir): # create output directory if it doesn't exist
        os.makedirs(output_dir)

    patient_vectors = [] # initialize dictionary to hold patient vectors
    patient_ids = []
    print("Start registration -->")

    # get postop ct nii file for each patient
    ct_files = glob.glob(os.path.join(base_dir, "**", "*postop_ct.nii"), recursive=True)
    ct_files = [f for f in ct_files if os.path.isfile(f)]

    # get segmentation mask nii file for each patient
    seg_masks = glob.glob(os.path.join(base_dir, "**", "*segmentation*.nii"), recursive=True)
    seg_masks = [f for f in seg_masks if os.path.isfile(f)]

    seg_map = {os.path.basename(seg).split("_")[0]: seg for seg in seg_masks} # map patient_id → segmentation

    if check_ct_seg_mapping(ct_files, seg_masks) == 0:
        print('WARNING: Mismatch in number of CT files and segmentation files.')
    
    for ct in ct_files: # loop through CT files
        patient_folder = os.path.basename(os.path.dirname(ct)) # get patient number from folder name

        new_name = f"{patient_folder}_postop_ct.nii" # add patient number to filename

        new_path = os.path.join(output_dir, new_name) # join directory and filename for new path
        
        patient_id = os.path.basename(patient_folder).split("_")[0]

        patient_folder_path = os.path.join(output_dir, patient_id)
        if not os.path.exists(patient_folder_path): # create output directory if it doesn't exist
            os.makedirs(patient_folder_path)

        output_ct_file = os.path.join(patient_folder_path, f"{patient_id}_registered_ct.nii")
         
        # set output file path
        if patient_id in seg_map:
            output_seg_file = os.path.join(patient_folder_path, f"{patient_id}_registered_seg.nii")
         
        # set atlas output file path
        if atlas_file:
            output_atlas_file = os.path.join(patient_folder_path, f"{patient_id}_atlas_in_patient.nii")

        # only register if orientations do not match (fix orientation if needed)
        if not ants.get_orientation(ants.image_read(ct)) == ants.get_orientation(ants.image_read(atlas_file)):
            print(f"Registering  and reorienting {patient_id}...")

            # register image
            Ogmi, mi = ants_registration(mni=mni_file,
                                        anat=ct,
                                        output_file=output_ct_file,
                                        atlas=atlas_file,
                                        atlas_output=output_atlas_file)
            
            mi_df.loc[len(mi_df)]= [patient_id, Ogmi, mi]
            
            # get overlay of registered image on mni and transformed brain atlas on patient image
            visualize_registration(
                fixed=ants.image_read(mni_file),
                moving=ants.image_read(ct),
                registered=ants.image_read(output_ct_file),
                atlas=ants.image_read(atlas_file),
                final_atlas=ants.image_read(output_atlas_file),
                patient_id=patient_id,
                output_dir=output_dir
            )
            # create binary brain region vector for each patient
            if patient_id in seg_map:
                vector = create_vector(seg_map[patient_id], output_atlas_file, atlas_labels_file)  # returns np.array
                patient_vectors.append(vector)
                patient_ids.append(patient_id) # append to list

            print(f"Registered {patient_id}. Image written to {output_ct_file}")

    # create vector matrix of brain region features
    df = pd.DataFrame(patient_vectors, 
                      columns=[f"Region_{i}" for i in range(patient_vectors[0].shape[0])])
    df.insert(0, "Patient_ID", patient_ids)
    # save vector to excel file
    df.to_excel(output_xlsx_path, index=False)
    print(f"Saved patient-region matrix → {output_xlsx_path}")

    print("Finished registration :))")
    return mi_df

def main():
    input = os.path.join(DATA_DIR, "segmentation-fixed-eric-V3/") # input image folder directory path
    output = os.path.join(OUTPUT_DIR, "registered_postop") # output registered image directory path
    output_xlsx_path = os.path.join(OUTPUT_DIR, "region_matrix_postop.xlsx") # brain region matrix excel path
    atlas = os.path.join(ATLAS_DIR, "Atlas_QSM.nii") # atlas path

    # atlas labels path
    atlas_labels = os.path.join(ATLAS_DIR, "Atlas_QSM.txt")

    # mni space image path
    mni = os.path.join(ATLAS_DIR, "mni.nii")
    
    # register images
    mi_df = multiple_img_registration(input_dir=input, 
                                        output_dir=output, 
                                        mni_file=mni, 
                                        atlas_file=atlas,
                                        output_xlsx_path=output_xlsx_path,
                                        atlas_labels_file=atlas_labels)

if __name__ == "__main__":
    main()
