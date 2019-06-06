import json
from google_images_download import google_images_download

#get list of labels (label idx -> label)
#get list of label:english names
#call google image api, for each english name(except "" and unk) create a folder and store 10 images
#for each folder, send them all though cnn, penultimate layer, stack them, average them, save stored idx
#in h5 with its org label idx



#download code
response = google_images_download.googleimagesdownload()

def downloadimages(query, output_dir, img_dir):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit":20,
                 "print_urls":False,
                 "output_directory": output_dir,
                 "image_directory": img_dir,
                 "size": "medium",
                 "aspect_ratio": "square"}
    try:
        response.download(arguments)

    # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit":20,
                     "print_urls":False,
                     "output_directory": output_dir,
                     "image_directory": img_dir,
                     "size": "medium"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass

# Driver Code

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument('--json_file', type=str, help='json file name')
    parser.add_argument('--output_dir', type=str, help='output_dir')
    args = parser.parse_args()

    with open(args.json_file) as f:
        labelid_english = json.load(f)

    for label_id, english_name in labelid_english.items():
        if english_name != 'no_english':
            downloadimages(english_name, args.output_dir, label_id)
            print()

if __name__ == "__main__":
    main()