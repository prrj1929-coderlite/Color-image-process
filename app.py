
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Color Image Processing",
    page_icon="🎨",
    layout="wide"
)

st.title("Color Image Processing Application")
st.write("Color Space Transformation | Segmentation | Object Detection | Visualization")

st.sidebar.header("Processing Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

category = st.sidebar.selectbox(
    "Select Category",
    [
        "Color Space Transformation",
        "Image Segmentation",
        "Object Detection",
        "Visualization"
    ]
)

operations = {
    "Color Space Transformation": [
        "RGB to Grayscale",
        "RGB to HSV",
        "RGB to Lab"
    ],
    "Image Segmentation": [
        "Otsu Thresholding"
    ],
    "Object Detection": [
        "Detect Red Objects",
        "Detect Green Objects",
        "Detect Blue Objects"
    ],
    "Visualization": [
        "RGB Channel Visualization",
        "HSV Channel Visualization"
    ]
}

operation = st.sidebar.selectbox(
    "Select Operation",
    operations[category]
)


# Normalize channels for display
def channel_display(channels):

    normalized = []

    for channel in channels:

        channel = channel.astype(np.float32)

        minimum = channel.min()
        maximum = channel.max()

        if maximum > minimum:
            channel = (channel - minimum) / (maximum - minimum)
        else:
            channel = np.zeros_like(channel)

        normalized.append(
            (channel * 255).astype(np.uint8)
        )

    return np.stack(normalized, axis=2)


# Create three-channel montage
def create_montage(channels):

    images = []

    for channel in channels:

        channel = cv2.normalize(
            channel,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        channel = cv2.cvtColor(
            channel,
            cv2.COLOR_GRAY2RGB
        )

        images.append(channel)

    return np.hstack(images)


# Object Detection Function
def detect_objects(image, color):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    if color == "Red":

        lower1 = np.array([0, 90, 50])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 90, 50])
        upper2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "Green":

        lower = np.array([35, 90, 50])
        upper = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

    else:

        lower = np.array([90, 90, 50])
        upper = np.array([135, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find connected components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    result = image.copy()
    object_count = 0

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area >= 30:

            object_count += 1

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                3
            )

    return result, mask, object_count


# Main Application
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image, dtype=np.uint8)

    st.subheader("Image Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            img,
            caption="Original Image",
            use_container_width=True
        )

    if st.sidebar.button("Apply Operation", type="primary"):

        result = None
        mask = None
        object_count = None
        channel_images = []
        channel_names = []

        # COLOR SPACE TRANSFORMATION
        if category == "Color Space Transformation":

            if operation == "RGB to Grayscale":

                result = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2GRAY
                )

            elif operation == "RGB to HSV":

                hsv = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2HSV
                )

                result = channel_display([
                    hsv[:, :, 0],
                    hsv[:, :, 1],
                    hsv[:, :, 2]
                ])

            elif operation == "RGB to Lab":

                lab = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2LAB
                )

                result = channel_display([
                    lab[:, :, 0],
                    lab[:, :, 1],
                    lab[:, :, 2]
                ])

        # IMAGE SEGMENTATION
        elif category == "Image Segmentation":

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY
            )

            threshold, result = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            st.info(
                f"Otsu Threshold Value: {threshold:.2f}"
            )

        # OBJECT DETECTION
        elif category == "Object Detection":

            color = operation.split()[1]

            result, mask, object_count = detect_objects(
                img,
                color
            )

        # VISUALIZATION
        elif category == "Visualization":

            if operation == "RGB Channel Visualization":

                channels = [
                    img[:, :, 0],
                    img[:, :, 1],
                    img[:, :, 2]
                ]

                channel_names = [
                    "Red Channel",
                    "Green Channel",
                    "Blue Channel"
                ]

            else:

                hsv = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2HSV
                )

                channels = [
                    hsv[:, :, 0],
                    hsv[:, :, 1],
                    hsv[:, :, 2]
                ]

                channel_names = [
                    "Hue Channel",
                    "Saturation Channel",
                    "Value Channel"
                ]

            channel_images = [
                np.repeat(
                    channel[:, :, None],
                    3,
                    axis=2
                )
                for channel in channels
            ]

            result = create_montage(channels)

        # Store results
        st.session_state["result"] = result
        st.session_state["mask"] = mask
        st.session_state["object_count"] = object_count
        st.session_state["channel_images"] = channel_images
        st.session_state["channel_names"] = channel_names
        st.session_state["operation_name"] = operation

    # Display Result
    if "result" in st.session_state:

        result = st.session_state["result"]

        with col2:

            st.image(
                result,
                caption=st.session_state["operation_name"],
                use_container_width=True
            )

        # Display object count
        if st.session_state["object_count"] is not None:

            st.success(
                "Total Objects Detected: "
                + str(st.session_state["object_count"])
            )

            with st.expander("View Binary Detection Mask"):

                st.image(
                    st.session_state["mask"],
                    caption="Binary Mask",
                    use_container_width=True
                )

        # Display individual channels
        if st.session_state["channel_images"]:

            st.subheader("Individual Channel Visualization")

            cols = st.columns(3)

            for i, channel_image in enumerate(
                st.session_state["channel_images"]
            ):

                with cols[i]:

                    st.image(
                        channel_image,
                        caption=st.session_state["channel_names"][i],
                        use_container_width=True
                    )

        # Download Result
        buffer = BytesIO()

        Image.fromarray(result).save(
            buffer,
            format="PNG"
        )

        st.download_button(
            "Download Processed Image",
            data=buffer.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )

else:

    st.info("Upload an image using the sidebar to get started.")

    st.subheader("Available Features")

    st.markdown("""
    1. **Color Space Transformation**
       - RGB to Grayscale
       - RGB to HSV
       - RGB to Lab

    2. **Image Segmentation**
       - Otsu Thresholding

    3. **Object Detection**
       - Red Object Detection
       - Green Object Detection
       - Blue Object Detection

    4. **Visualization**
       - RGB Channel Visualization
       - HSV Channel Visualization
    """)
