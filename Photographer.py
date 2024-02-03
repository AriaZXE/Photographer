import cv2
import os
import keyboard
import time

def get_last_image_index(folder_name):
    index_file = os.path.join(folder_name, "last_index.txt")
    if os.path.exists(index_file):
        with open(index_file, "r") as file:
            last_index = int(file.read())
        return last_index
    else:
        return 0

def update_last_image_index(folder_name, last_index):
    index_file = os.path.join(folder_name, "last_index.txt")
    with open(index_file, "w") as file:
        file.write(str(last_index))

def capture_and_save_images():
    folder_name = "images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    last_index = get_last_image_index(folder_name)
    img_index = last_index

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam Output', frame)

        if keyboard.is_pressed('s'):
            file_name = f"{folder_name}/img{img_index}.png"
            cv2.imwrite(file_name, frame)
            print(f"Image successfully saved: {file_name}")
            img_index += 1
            update_last_image_index(folder_name, img_index)

            while keyboard.is_pressed('s'):
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_images()



# by Aria 🐸