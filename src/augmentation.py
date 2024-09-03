import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='dataset')
    parser.add_argument('image_output', '-i', type=str, default='augmented_images')

def load_image(data_dir, image_file):
    return cv2.imread(os.path.join(data_dir, image_file))

def random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def augmentation(image, steering_angle, output_dir, filename_prefix):
    image, steering_angle = random_flip(image, steering_angle)

    # Save augmented image
    filename = os.path.join(output_dir, f"{filename_prefix}.jpg").replace('\\', '/')
    cv2.imwrite(filename, image)
    
    return filename, steering_angle

def draw_distribution(dataset_path):
    orig_data_df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # Get the path to camera center, left, right
    X = orig_data_df[['center', 'left', 'right']].values
    # Get drive's steering
    y = orig_data_df['steering'].values

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the original distribution on the left subplot
    axs[0].hist(y, bins=30, color='blue', edgecolor='black')
    axs[0].set_title("Distribution Before Augmentation")
    axs[0].set_xlabel("Steering Angle")
    axs[0].set_ylabel("Frequency")

    augmented_data_df = pd.read_csv(os.path.join(dataset_path, 'augmented_driving_log.csv'))

    # Get the path to camera center
    X_augmented = augmented_data_df['image'].values
    # Get drive's steering
    y_augmented = augmented_data_df['steering'].values

    # Plot the augmented distribution on the right subplot
    axs[1].hist(y_augmented, bins=30, color='green', edgecolor='black')
    axs[1].set_title("Distribution After Augmentation")
    axs[1].set_xlabel("Steering Angle")
    axs[1].set_ylabel("Frequency")

    # Save the subplots
    plt.tight_layout()
    plt.savefig('distribution_data.jpg')

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.image_output, exist_ok=True)
    
    csv_file_path = os.path.join('args.data_dir', 'driving_log.csv')
    fake_df = pd.read_csv(csv_file_path, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    new_data = []

    for idx, row in fake_df.iterrows():
        center, left, right, steering, throttle, reverse, speed = row
        
        # Save the original image with the original steering angle
        image = load_image(args.data_dir, center)
        base_filename = f"{os.path.splitext(os.path.basename(center))[0]}"    
        orig_filename = f"{base_filename}.jpg"
        orig_filepath = os.path.join(args.image_output, orig_filename).replace('\\', '/')
        cv2.imwrite(orig_filepath, image)
        new_data.append([orig_filepath, steering, throttle, reverse, speed])
        
        # Randomly select one image from center camera and perform augmentation
        if steering < 0 and np.random.rand() < 0.5:
            augmented_image_file = center
            augmented_steering = steering
        
            image = load_image(args.data_dir, augmented_image_file)
            base_filename = f"{os.path.splitext(os.path.basename(augmented_image_file))[0]}"
        
            # Perform augmentation once
            aug_filename, aug_steering = augmentation(image, augmented_steering, args.image_output, f"{base_filename}_aug")
            new_data.append([aug_filename, aug_steering, throttle, reverse, speed])
        
                
    # Save new CSV
    new_df = pd.DataFrame(new_data, columns=['image', 'steering', 'throttle', 'reverse', 'speed'])
    new_csv_path = os.path.join('args.data_dir', 'augmented_driving_log.csv')
    new_df.to_csv(new_csv_path, index=False)
    draw_distribution(dataset_path=args.data_dir)