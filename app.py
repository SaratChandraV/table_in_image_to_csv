import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import cv2 as cv
import pytesseract

min_size_of_cell = st.sidebar.slider('Min. size of Cell', 1, 50000, 5000)
st.sidebar.write("Adjust this setting so that no text gets selected and not too large that any cell will be missed.")
table_contour_factor = st.sidebar.slider('Table Contour Factor', 1, 100, 10)
st.sidebar.write("Adjust this setting so that the border of entire table / image is not selected. Also not too large that any cell will be missed.")

if 'significant_contour_list' not in st.session_state:
    st.session_state.significant_contour_list = []

if 'imgray' not in st.session_state:
    st.session_state.imgray = 0

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

def remove_newline_char(a):
    if a == "":
        return ""
    if str(a) == "NaN":
        return ""
    if a[-1] == '\n':
        return a[:-1]
    return a

def convert_DF_to_csv(df):
    s = ""
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if j == df.shape[1] - 1:
                s = s + str(df.iloc[i,j])
            else:
                s = s + str(df.iloc[i,j]) + ","
                
        s = s + '\n'
    return s

def runalgo():
    
    # now for easy of computing and establishing regions for text mining each signifiant contour, their respective bounding rectangular boxes are found.
    significant_contour_list = st.session_state.significant_contour_list
    significant_contour_rect_details = []
    imgray = st.session_state.imgray
    for i in range(0,len(significant_contour_list)):
        significant_contour_rect_details.append(cv.boundingRect(significant_contour_list[i]))

    # the center of each rect for each cell is computed to further easy in sorting and finding the order of cells.
    significant_contour_rect_center = []
    for i in range(0,len(significant_contour_rect_details)):
        significant_contour_rect_center.append((significant_contour_rect_details[i][0] + 
                                                significant_contour_rect_details[i][2] / 2,
                                            significant_contour_rect_details[i][1] + 
                                                significant_contour_rect_details[i][3] / 2,
                                            i))
        
    # since the order of contours can be different and the exact no. of rows and columns are always unclear
    # 1. the contour with least y value is found
    # 2. then the header row is figured out by comparing the y value of each cell with the least y value
    # 3. still the header row may not be in a correct sequence hence they are ordered by x value to represent the header row of a flat table.
    unordered_header_rows = []
    min_y = 1000000.0
    min_index = 0
    for i in range(0,len(significant_contour_rect_center)):
        if min_y >= significant_contour_rect_center[i][1]:
            min_y = significant_contour_rect_center[i][1]
            min_index = i
    for i in range(0,len(significant_contour_rect_center)):
        if abs(min_y - significant_contour_rect_center[i][1]) <= 5:
            unordered_header_rows.append(i)
    header_rows_x_values_unordered = []
    for i in range(0,len(unordered_header_rows)):
        header_rows_x_values_unordered.append(significant_contour_rect_center[unordered_header_rows[i]][0])
    header_rows_x_values_index = np.argsort(header_rows_x_values_unordered)
    header_rows_index = []
    for i in range(0,len(header_rows_x_values_index)):
        header_rows_index.append(unordered_header_rows[header_rows_x_values_index[i]])

    # now from ordered header row cells the remaining cells that are vertically below are found out and then they are ordered by y value.
    table_cells_index = []
    for i in header_rows_index:
        table_cells_index.append([i])
    for i in range(0,len(header_rows_index)):
        for j in range(0,len(significant_contour_rect_center)):
            if abs(significant_contour_rect_center[j][0] - 
                significant_contour_rect_center[header_rows_index[i]][0]) <= 5 and j != header_rows_index[i]:
                table_cells_index[i].append(j)
    for i in range(0,len(header_rows_index)):
        a = list(table_cells_index[i][1:])
        col_y = []
        for j in a:
            col_y.append(significant_contour_rect_center[j][1])
        col_y_index = np.argsort(col_y)
        col_y_index = col_y_index
        b = []
        for j in col_y_index:
            b.append(a[j])
        table_cells_index[i] = [header_rows_index[i]] + b

    # for ech cell tesseract is used to extract the text and stored in a 2d list.
    # pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract" #this is must for macOS M1
    table_contents = []
    for i in range(0,len(table_cells_index)):
        a = []
        for j in table_cells_index[i]:
            y = significant_contour_rect_details[j][1]
            h = significant_contour_rect_details[j][3]
            x = significant_contour_rect_details[j][0]
            w = significant_contour_rect_details[j][2]
            cropped = imgray[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            a.append(text)
        table_contents.append(a)
    df = pd.DataFrame(table_contents)
    df = df.transpose() # since the data is column wise we have to apply transpose to convert to a flat table.
    # some preprocessing is required like removing new line character at the last for each cell in the dataframe.
    for i in range(0,len(df.columns)):
        df[i] = df.apply(lambda x: remove_newline_char(x[i]),axis = 1)
    st.session_state.df = df
        
    

def contour_area(a):
    return cv.contourArea(a)


def setCountours(img_bytes):
    
    imgray = cv.cvtColor(img_bytes, cv.COLOR_BGR2GRAY)
    st.session_state.imgray = imgray
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # creating a list of areas by each contour
    contour_area_list = []
    for i in range(0,len(contours)):
        contour_area_list.append(contour_area(contours[i]))
    contour_area_list = np.array(contour_area_list)

    # finding only significant_counters -- here the area is used as metric to eliminate text contours and other small regions
    significant_contour_list = []
    max_contour_area = max(contour_area_list)
    for i in range(0,len(contours)):
        # here it is assumed that each cell int able be atleast 800 sq. pixels
        # there is always a possiblity of non exact crop of image hence there will always be atleast 1 large contour around the table border.
        if contour_area_list[i] > min_size_of_cell and contour_area_list[i] < max_contour_area / table_contour_factor:
            significant_contour_list.append(contours[i])
    significant_contour_list = np.array(significant_contour_list)
    st.session_state.significant_contour_list = significant_contour_list
    im_contours_significant = img_bytes.copy()
    im_contours_significant = cv.drawContours(im_contours_significant, significant_contour_list, -1, (0,255,0), 3) # the contours are set to be visible in green

    img = cv.cvtColor(im_contours_significant, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil

def convertImg(img):
    nparr = np.array(img.convert('RGB'))
    return nparr[:, :, ::-1].copy()

st.title("Table from Image using opencv")

image = Image.open('sports_data.png')
image_contoured = setCountours(convertImg(image))

info_placeholder = st.empty()

# tab1, tab2 = st.tabs(["Data","Contoured Image"])

# upload_image_button = st.button("Upload Image")

uploaded_file = st.file_uploader("Upload Image",type=['png'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    image_contoured = setCountours(convertImg(image))


st.sidebar.header("Original Image")
st.sidebar.image(image)

col_b_1, col_b_2 = st.columns(2)

with col_b_1:
    st.button("Convert",on_click=runalgo)

with col_b_2:
    st.download_button('Download CSV', convert_DF_to_csv(st.session_state.df), file_name='data.csv')

col1, col2 = st.columns(2)

with col2:
    st.header("Data")
    if st.session_state.df.shape[0] != 0:
        st.dataframe(st.session_state.df)

with col1:
    st.header("Image with Contours")
    st.image(image_contoured)