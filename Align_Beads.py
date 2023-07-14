import cv2
import numpy as np

# Directory with beads images
beads_file = "/Volumes/Candace A/CandaceImagingData/20230623_4monthPostIV_SiRHalo/Beads_2023-06-23_15-06-54"

def align_images(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect features (e.g., keypoints) in the images
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match the keypoints between the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate the transformation matrix using RANSAC
    M, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Apply the transformation to image2 to align it with image1
    aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

    return aligned_image2, M

# Read the two input images
image1 = cv2.imread(beads_file + 'beads_647_0.tif')
image2 = cv2.imread(beads_file + 'beads_488_0.tif')

# Align the images and obtain the transformation matrix
transformed_image2, transformation_matrix = align_images(image1, image2)

# Specify the output path and filename for the transformed image
output_path1 = beads_file + 'beads_647.tif'
output_path2 = beads_file + 'beads_488.tif'
output_path_transformed = beads_file + 'transformed_image2.tif'

# Specify the output path and filename for the transformation matrix
matrix_output_path = beads_file + 'transformation_matrix.txt'

# Save the original images
cv2.imwrite(output_path1, image1)
cv2.imwrite(output_path2, image2)

# Save the transformed image
cv2.imwrite(output_path_transformed, transformed_image2)

# Save the transformation matrix
np.savetxt(matrix_output_path, transformation_matrix)

# Display a success message
#cv2.imshow('red', image1)
#cv2.imshow('green', image2)
#cv2.imshow('Transformed Image 2', transformed_image2)
print(f"Transformed image 2 saved to: {output_path_transformed}")
print(f"Transformation matrix saved to: {matrix_output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
