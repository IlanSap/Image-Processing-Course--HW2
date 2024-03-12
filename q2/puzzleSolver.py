import cv2
import numpy as np
import os
import shutil

# input: a non square matrix
# output: square matrix with padding
def make_affine_matrix_square(affine_matrix):
    # Pad the nxm matrix to make it a nxn square matrix
    return np.vstack([affine_matrix, [0, 0, 1]])


# input: two arrays of points- source points and dst points, the image we want to stitch to canvas
# output: affine transformation matrix
def get_Affine_transform(src_points, dst_points):

        # Calculate the affine transformation matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        height, width = affine_matrix.shape

        # padding affine matrix to be square
        if(height != width):
           affine_matrix = make_affine_matrix_square(affine_matrix)

        return affine_matrix


# input: two arrays of points- source points and dst points, the image we want to stitch to canvas
# output: projective transformation matrix
def get_projective_transform(src_points, dst_points):
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return perspective_matrix


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches,is_affine):

    src_points, dst_points = matches[:, 0], matches[:, 1]

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    if(is_affine):
        transform_image = get_Affine_transform(src_points, dst_points)
    else:
        transform_image = get_projective_transform(src_points, dst_points)
    return transform_image


def inverse_transform_target_image(target_img, original_transform, canvas_size):
    # Calculate the inverse transformation matrix
    inverse_transform_matrix = np.linalg.inv(original_transform)

    # Perform inverse transformation using warpPerspective
    inverse_transformed_image = cv2.warpPerspective(target_img, inverse_transform_matrix, (canvas_size[1], canvas_size[0]))
    return inverse_transformed_image


# input: two images- source image is the first image and the other one is every another picture after transformation
# output: one picture includes the two picture stitched
def stitch(source_img, inverse_transform_image):

    stitched_image=cv2.max(source_img, inverse_transform_image)
    return stitched_image


def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']

    # for every puzzle:
    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # get the first image (the canvas)
        onePiecePath = os.path.join(pieces_pth, f'piece_1.jpg')
        source_img = cv2.imread(onePiecePath)

        # get canvas size (first image size)
        canvas_height, canvas_width, channels = source_img.shape
        canvas_size = (canvas_height, canvas_width)

        # insert the first image to absolute images file
        cv2.imwrite(os.path.join(edited, f'piece_1_absolute.jpg'), source_img)

        # for every image from the second one to the last one:
        for i in range(2, n_images+1):
            onePiecePath = os.path.join(pieces_pth, f'piece_{i}.jpg')
            target_img = cv2.imread(onePiecePath)
            original_transform = get_transform(matches[i-2], is_affine)
            inverse_transform_image = inverse_transform_target_image(target_img, original_transform, canvas_size)

            inverse_transform_image_name = f'piece_{i}_absolute.jpg'
            cv2.imwrite(os.path.join(edited, inverse_transform_image_name), inverse_transform_image)

            source_img = stitch(source_img, inverse_transform_image)

        # an option to see the final puzzle:
        cv2.imshow('fullPuzzle', source_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # create image's file to the final puzzle
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), source_img)
