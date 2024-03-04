# make imports
import csv
import os
import requests

# read the csv file
def read_csv(file_name):
    
    # check if the file exists
    if os.path.exists(file_name):

        # open the file
        with open(file_name, 'r') as file:

            # read the file
            reader = csv.reader(file)

            # return the data
            return list(reader)
        
    return None

# download the images and store them in a directory
def download_images(data, directory):

    # maintain a download count to see the number of images downloaded
    download_count = 0
    
    # check if the directory exists
    if not os.path.exists(directory):

        # create the directory
        os.mkdir(directory)
        
    # loop through the data
    for row in data:
        
        # get the product id and create a dir corresponding to each product
        image_name = row[0]
        
        # create the directory
        product_directory = os.path.join(directory, image_name)
        
        # check if the directory exists
        if not os.path.exists(product_directory):

            # create the directory
            os.mkdir(product_directory)

        # get the list of image links and remove the brackets
        image_links = row[1][1:-1]

        """
        download the images and store them
        in the directory corresponding to the image name
        """

        # loop through the image links
        for index, link in enumerate(image_links.split(',')):

            # get the image path
            image_path = os.path.join(product_directory, f"{image_name}_{index+1}.jpg")

            # download the image
            with open(image_path, 'wb') as file:
                
                """
                strip the whitespaces
                get rid of the quotes in the link
                and get the image content
                """

                image_content = requests.get(link.strip()[1:-1]).content
                # write the image content to the file
                file.write(image_content)

                # increment download count if image downloaded successfully
                if (os.path.exists(image_path)):
                    download_count += 1

    return download_count


# read the csv file
num_image_review = read_csv('A2_Data.csv')

# remove the first row (fieldnames)
num_image_review.pop(0)

print(len(num_image_review))

# download the images
download_count = download_images(num_image_review, 'images')

# check if all images have beeen downloaded
image_count = 0
for row in num_image_review:
    image_count += len(row[1][1:-1].split(','))

if (image_count == download_count):
    print(f"Downloaded {download_count} images successfully")