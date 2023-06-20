import cv2
import os

def load_images_from_folder(folder):
    # images = []
    for filename in os.listdir(folder):
        print("Filename: ", filename)
        img = cv2.imread(os.path.join(folder,filename))

        tempTuple = os.path.splitext(filename)
        filename = tempTuple[0]

        # Crop into four quadrants (1-top left, 2-top right, 3- bottom left, 4- bottom right)
        (height, width, _) = img.shape
        
        crop_1 = img[0:int(height/2), 0:int(width/2)]
        crop_2 = img[0:int(height/2), int(width/2)+1:width]
        crop_3 = img[int(height/2)+1:height, 0:int(width/2)]
        crop_4 = img[int(height/2)+1:height, int(width/2)+1:width]

        folder_1 = "/media/asathyam/Media/spot-veg/4_dataset_cropped/1_top_left"
        folder_2 = "/media/asathyam/Media/spot-veg/4_dataset_cropped/2_top_right"
        folder_3 = "/media/asathyam/Media/spot-veg/4_dataset_cropped/3_bottom_left"
        folder_4 = "/media/asathyam/Media/spot-veg/4_dataset_cropped/4_bottom_right"
        
        crop_1_name = folder_1 + "/" + filename + "_1.png"
        crop_2_name = folder_2 + "/" + filename + "_2.png"
        crop_3_name = folder_3 + "/" + filename + "_3.png"
        crop_4_name = folder_4 + "/" + filename + "_4.png"

    

        cv2.imwrite(crop_1_name, crop_1)
        cv2.imwrite(crop_2_name, crop_2)
        cv2.imwrite(crop_3_name, crop_3)
        cv2.imwrite(crop_4_name, crop_4)


        # cv2.imshow('cv_img_1', crop_1)
        # cv2.imshow('cv_img_2', crop_2)
        # cv2.imshow('cv_img_3', crop_3)
        # cv2.imshow('cv_img_4', crop_4)
        cv2.imshow('cv_img', img)
        cv2.waitKey(100)


if __name__ == "__main__":
    # folder = "/media/asathyam/Media/spot-veg/dataset_lite/bushes-lite"
    # folder = "/media/asathyam/Media/spot-veg/dataset_lite/grass-dense-lite"
    # folder = "/media/asathyam/Media/spot-veg/dataset_lite/grass-sparse-lite"
    # folder = "/media/asathyam/Media/spot-veg/dataset_lite/trees-lite"
    folder = "/media/asathyam/Media/spot-veg/3_dataset_lite/umd_multi_6_lite"
    load_images_from_folder(folder)

    cv2.destroyAllWindows()