import cv2
import os
from cv2 import aruco
from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C


# img = cv2.imread("charuco_board.jpg")
# possible_dicts = [
#     aruco.DICT_4X4_50, aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
#     aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
#     aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
#     aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
#     aruco.DICT_ARUCO_ORIGINAL
# ]

# for d in possible_dicts:
#     aruco_dict = aruco.getPredefinedDictionary(d)
#     corners, ids, rejected = aruco.detectMarkers(img, aruco_dict)
#     print(f"Dictionary {d}: {len(corners)} markers detected")



# ====================== Test interpolated ===================


folder_path = "dataset/left"
files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
files.sort()   # works fine since filenames are zero-padded (0000, 0001, ...)

for filename in files:
    # img = cv2.imread("dataset/left/img_0660.jpg", cv2.IMREAD_COLOR)
    img = cv2.imread(f"{folder_path}/{filename}", cv2.IMREAD_COLOR)

    # # Define the target resolution (width, height)
    # target_resolution = (368, 272)  # For example: 800x600
    # downscaled_image = cv2.resize(img, target_resolution, interpolation=cv2.INTER_AREA)
    # cv2.imwrite('downscaled_image.jpg', downscaled_image)
    # breakpoint()


    # img_l = cv2.imread("/home/robotlearning2/stereo-calib/dataset/left/img_0000.jpg", cv2.IMREAD_COLOR)
    gray_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(C.ARUCO_DICT)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Define the board — must match the actual printed one
    board = CharucoBoard(charuco_data=CharucoBoardData(aruco_dict=C.ARUCO_DICT,
                                                                squares_vertically=C.SQUARES_VERTICALLY,
                                                                squares_horizontally=C.SQUARES_HORIZONTALLY,
                                                                square_length=C.SQUARE_LENGTH,
                                                                marker_length=C.MARKER_LENGTH))

    # Detect markers
    corners_l, ids_l, _ = aruco_detector.detectMarkers(gray_l)
    print("filename, Number of detected corners: ", filename, len(corners_l))
    # print("ids_l: ", ids_l)
    # print("corners_l: ", corners_l)
    # print("Board marker IDs:", board.board.getIds().ravel())
    # breakpoint()

    # # Optional: Draw and show
    vis_img = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
    # Draw detected markers
    cv2.aruco.drawDetectedMarkers(vis_img, corners_l, ids_l)

    # Interpolate ChArUco corners
    if len(corners_l) > 0:
        retval_l, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners_l,
            markerIds=ids_l,
            image=gray_l,
            board=board.board
        )
        print(f"Interpolated {retval_l} ChArUco corners.")

        if retval_l > 0:
            # Draw Charuco corners (red circles with IDs)
            for i, corner in enumerate(charuco_corners_l):
                c = tuple(corner[0].astype(int))
                cv2.circle(vis_img, c, 5, (0, 0, 255), -1)  # red dot
                cv2.putText(vis_img, str(charuco_ids_l[i][0]), (c[0]+5, c[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    
    # breakpoint()
    # Show the result
    # cv2.imshow("Detected ArUco markers", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
