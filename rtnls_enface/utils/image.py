import cv2


def can_match_resolution(target, image_b):
    # Get dimensions of both images
    width_a, height_a = target.shape
    width_b, height_b = image_b.shape

    # Calculate aspect ratios
    aspect_ratio_b = width_b / height_b

    # Resize image B to match the height or width of image A while maintaining its aspect ratio
    new_height_b = round(width_a / aspect_ratio_b)
    new_size_b = (width_a, new_height_b)

    # Compare resized dimensions of image B with dimensions of image A
    return new_size_b == (width_a, height_a)


def match_resolution(image, target_resolution, interpolation=cv2.INTER_CUBIC):
    # Get dimensions of both images
    width_target, height_target = target_resolution[0], target_resolution[1]
    width, height = image.shape[1], image.shape[0]

    if width_target == width and height_target == height:
        return image

    # Calculate aspect ratios
    aspect_ratio_b = width / height

    # Resize image B to match the height or width of image A while maintaining its aspect ratio
    new_height = round(width_target / aspect_ratio_b)
    new_size = (width_target, new_height)

    # Compare resized dimensions of image B with dimensions of image A
    if new_size != target_resolution:
        raise Exception(
            f"Image with shape {image.shape} cannot match resolution {target_resolution}. Resized resolution would be {new_size}"
        )

    new_image = cv2.resize(image, new_size, interpolation=interpolation)
    return new_image
