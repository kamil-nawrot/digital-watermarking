import logging

import attacks.Attacks as attacks

logging.basicConfig(level=logging.DEBUG)

original_im_path = 'images\\mandrill_512.jpg'
original_wm_path = 'images\\lenna_512.jpg'

original_im = []
original_wm = []
images_with_watermarks = []  # create_images_with_watermarks(filename)


def perform_attacks_for_single_image(im):
    attacked_images = attacks.perform_all_attacks_on_watermarked_image(im)
    psnrs = []
    for atck_im in attacked_images:
        extracted_wm = extract_watermark(atck_im)
        psnrs.append(attacks.check_psnr(original_wm, extracted_wm))
    return psnrs


# method 1
image_with_watermark = []
psnrs_method1 = perform_attacks_for_single_image(image_with_watermark)

# method 2

# method 3

# method ...
