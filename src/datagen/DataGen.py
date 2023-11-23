import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Frames per second
    success, image = vidcap.read()
    count = 0

    while success:
        # Save frame every second
        if count % int(fps) == 0:
            frame_file = os.path.join(output_folder, f"frame_{count // int(fps)}.jpg")
            cv2.imwrite(frame_file, image)     # save frame as JPEG file
            print(f"Saved {frame_file}")

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print("Done extracting Frames.")

def main():
    # Example usage
    video_path = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos/video1.mp4'  # Replace with your video path
    output_folder = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Frames'     # Replace with your desired output folder
    extract_frames(video_path, output_folder)

if __name__ == "__main__":
    main()
