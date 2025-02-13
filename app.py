import streamlit as st
from streamlit_labelstudio import st_labelstudio
import cv2
from alignment.alignment import load_defect_pair, get_diff_map
from frontend.image_processor import numpy_to_base64, load_images_from_json, get_image_pair_for_studio_input, apply_colormap, conv_kernel
from frontend.label_config import config, interfaces, user, two_row_config
from prelabel.prelabel import task_generator
from frontend.read_lrf import get_defect_list, get_lrf_file
from backend.sqlite_functions import save_json_to_sqlite, fetch_results
from frontend.json_functions import save_results_to_json, sync_labels_across_3images, get_area_size
import sqlite3
import os
import traceback
import argparse
from loguru import logger

# modify the .db filename if you want a fresh start (will create file for you)
db_path = "backend/db_files/the_regional_dataset.db"
# pass in trained model checkpoint (currently only supports YOLO)
model_path = "/home/hubert007/Code/label_tool/labeling/runs/detect/megamind_nojit_vmin_fix/weights/best.pt"

st.set_page_config(layout='wide')

# Initialize session state for results_raw and image index
if 'previous_results_raw' not in st.session_state:
    st.session_state.previous_results_raw = None
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'lot_index' not in st.session_state:
    st.session_state.lot_index = 0
# MODIFIED: 新增 update_pressed 状态，用于 Update 按钮的处理
if 'update_pressed' not in st.session_state:
    st.session_state.update_pressed = False

# Function to check if results_raw has changed
def has_results_raw_changed(current_results_raw):
    previous_results_raw = st.session_state.previous_results_raw
    if previous_results_raw != current_results_raw:
        st.session_state.previous_results_raw = current_results_raw
        return True
    return False

# Initialize (or create) the database table if not exists
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS results_table (
    id INTEGER PRIMARY KEY,
    image_path TEXT,
    results_json TEXT
)
''')
conn.commit()
conn.close()

# Sidebar for directory-like structure
st.sidebar.title('Image Directory')

data_dir = "/mnt/fs0/dataset/Layer_M"  
items = os.listdir(data_dir)

# Filter out only directories (exclude 'log')
lot_ids = [item for item in items if os.path.isdir(os.path.join(data_dir, item)) and item != 'log']
lot_ids.sort()
try:
    selected_lot_id = st.sidebar.selectbox('Select Lot ID', lot_ids, index=st.session_state.lot_index)
except:
    selected_lot_id = st.sidebar.selectbox('Select Lot ID', lot_ids, index=0)

lrf_file = get_lrf_file(data_dir, selected_lot_id)
defect_images_id_list, defect_type = get_defect_list(lrf_file)

# Previous Image button
if st.sidebar.button('Previous Image'):
    st.session_state.image_index -= 1
    if st.session_state.image_index < 0:
        st.session_state.lot_index = (st.session_state.lot_index - 1) % len(lot_ids)
        selected_lot_id = lot_ids[st.session_state.lot_index]
        lrf_file = get_lrf_file(data_dir, selected_lot_id)
        defect_images_id_list, defect_type = get_defect_list(lrf_file)
        st.session_state.image_index = len(defect_images_id_list) - 1

# Next Image button
if st.sidebar.button('Next Image'):
    st.session_state.image_index += 1
    if st.session_state.image_index >= len(defect_images_id_list):
        st.session_state.image_index = 0
        st.session_state.lot_index = (st.session_state.lot_index + 1) % len(lot_ids)
        selected_lot_id = lot_ids[st.session_state.lot_index]
        lrf_file = get_lrf_file(data_dir, selected_lot_id)
        defect_images_id_list, defect_type = get_defect_list(lrf_file)

# MODIFIED: 增加 Update 按钮，允许用户在提交后继续编辑当前图片
if st.sidebar.button('Update'):
    st.session_state.update_pressed = True
    st.write("Update button pressed, staying on the current image.")
    st.rerun()

try:
    selected_image_id = st.sidebar.selectbox('Select Image ID', defect_images_id_list, index=st.session_state.image_index)
except:
    selected_image_id = '1'
st.session_state.image_index = defect_images_id_list.index(selected_image_id)
st.write(f"defect type : {defect_type[st.session_state.image_index]}")

prelabel_methods = ['minmax', 'YOLO']
selected_prelabel_method = st.sidebar.selectbox('Select prelabel method', prelabel_methods, index=0)

alignment_methods = ['SIFT', 'Correlate']
selected_alignment_method = st.sidebar.selectbox('Select alignment method', alignment_methods, index=0)

# Slider for saturation adjustment
vmin_level = st.sidebar.slider('vmin_level', -2.0, 0.0, -0.5)
max_features = st.sidebar.slider('max_features', 20.0, 1000.0, 1000.0, 20.0)
max_shift = st.sidebar.slider('max_shift', 0.0, 20.0, 10.0)
ransac_reproj_threshold = st.sidebar.slider('ransac_reproj_threshold', 0.0, 0.3, 0.10, 0.01)
crop_size = st.sidebar.slider('crop_size', 1.0, 30.0, 15.0, 1.0)
conv_kernel_size = st.sidebar.slider('conv_kernel_size', 1.0, 10.0, 2.0, 1.0)

T_images, T_metadata = get_image_pair_for_studio_input(
    data_dir, selected_lot_id, selected_image_id, "T", vmin_level, max_features, max_shift,
    ransac_reproj_threshold, selected_alignment_method, conv_kernel_size
)
Rt_images, Rt_metadata = get_image_pair_for_studio_input(
    data_dir, selected_lot_id, selected_image_id, "Rt", vmin_level, max_features, max_shift,
    ransac_reproj_threshold, selected_alignment_method, conv_kernel_size
)

images_base64 = [T_images[0], T_images[1], T_images[2], Rt_images[0], Rt_images[1], Rt_images[2]]
metadatas = [T_metadata, Rt_metadata]

config = two_row_config

img_path = f"{data_dir}, {selected_lot_id}, {selected_image_id}"
existing_labels = fetch_results(img_path, db_path)

if not existing_labels:
    results_raw = st_labelstudio(
        config, interfaces, user,
        task_generator(images_base64, crop_size, metadatas=metadatas, method=selected_prelabel_method, model_path=model_path)
    )
else:
    print("Labels already exist. Using existing Labels.")
    st.write("Using existing labels.")
    results_raw = st_labelstudio(
        config, interfaces, user,
        task_generator(images_base64, crop_size, method=selected_prelabel_method, existing_labels=existing_labels)
    )

if results_raw is not None and has_results_raw_changed(results_raw):
    results_raw = sync_labels_across_3images(results_raw)
    # 保存標注結果到數據庫
    save_json_to_sqlite(img_path, results_raw, db_path)
    # MODIFIED: 如果用戶沒有點擊 Update 按鈕，則自動切換到下一張圖片；若 Update 被點擊，保持當前圖片
    if not st.session_state.update_pressed:
        st.session_state.image_index += 1
        if st.session_state.image_index >= len(defect_images_id_list):
            st.session_state.image_index = 0
            st.session_state.lot_index = (st.session_state.lot_index + 1) % len(lot_ids)
            selected_lot_id = lot_ids[st.session_state.lot_index]
            lrf_file = get_lrf_file(data_dir, selected_lot_id)
            defect_images_id_list, defect_type = get_defect_list(lrf_file)
    # 重置 update_pressed 標誌
    st.session_state.update_pressed = False
    st.rerun()





##################################################################
修改部分说明
新增 Update 按钮
在侧边栏添加了如下代碼：

python
複製
if st.sidebar.button('Update'):
    st.session_state.update_pressed = True
    st.write("Update button pressed, staying on the current image.")
    st.rerun()
這段代碼確保當用戶點擊 Update 按鈕時，系統記錄下來並立即重跑頁面，但不改變 image_index，使得用戶能夠在同一張圖片上進行更新。

修改提交標注後的邏輯
在提交結果的 if 區塊中，增加了判斷：

python
複製
if not st.session_state.update_pressed:
    st.session_state.image_index += 1
    ...
表示如果 Update 按鈕未被點擊，則自動將 image_index 自增進入下一張圖片；如果 Update 被點擊，則保持當前圖片不變。

重置 update_pressed 標誌
在提交結果後，將 st.session_state.update_pressed 重置為 False，保證下次進入正常流程。




下面我将逐行解释这段代码的逻辑：

python
複製
if results_raw is not None and has_results_raw_changed(results_raw):
    results_raw = sync_labels_across_3images(results_raw)
    # 保存標注結果到數據庫
    save_json_to_sqlite(img_path, results_raw, db_path)
    # MODIFIED: 如果用戶沒有點擊 Update 按鈕，則自動切換到下一張圖片；若 Update 被點擊，保持當前圖片
    if not st.session_state.update_pressed:
        st.session_state.image_index += 1
        if st.session_state.image_index >= len(defect_images_id_list):
            st.session_state.image_index = 0
            st.session_state.lot_index = (st.session_state.lot_index + 1) % len(lot_ids)
            selected_lot_id = lot_ids[st.session_state.lot_index]
            lrf_file = get_lrf_file(data_dir, selected_lot_id)
            defect_images_id_list, defect_type = get_defect_list(lrf_file)
    # 重置 update_pressed 標誌
    st.session_state.update_pressed = False
逐步解釋
檢查是否有新提交的標注結果

python
複製
if results_raw is not None and has_results_raw_changed(results_raw):
results_raw is not None：確認前端返回了標注結果（results_raw 非空）。
has_results_raw_changed(results_raw)：檢查此次提交的標注結果是否與上次不同。如果結果有改變，則返回 True。
同步多個視圖的標注信息

python
複製
results_raw = sync_labels_across_3images(results_raw)
這一行會將 results_raw 中的標注信息在不同圖片（如 image1、image2、image3 或其他）之間進行同步，確保標注結果一致。
例如，如果某個區域在 image3 上標注了，可能需要自動同步到其他相關圖片中。
保存最新標注到數據庫

python
複製
save_json_to_sqlite(img_path, results_raw, db_path)
將當前圖片（由 img_path 唯一標識）和最新的標注結果 results_raw 保存到 SQLite 數據庫中，確保數據持久化。
根據 Update 按鈕決定是否切換圖片

python
複製
if not st.session_state.update_pressed:
    st.session_state.image_index += 1
    if st.session_state.image_index >= len(defect_images_id_list):
        st.session_state.image_index = 0
        st.session_state.lot_index = (st.session_state.lot_index + 1) % len(lot_ids)
        selected_lot_id = lot_ids[st.session_state.lot_index]
        lrf_file = get_lrf_file(data_dir, selected_lot_id)
        defect_images_id_list, defect_type = get_defect_list(lrf_file)
if not st.session_state.update_pressed:
如果用戶沒有點擊 Update 按鈕（即 update_pressed 為 False），則系統認為用戶希望提交標注後自動進入下一張圖片。
st.session_state.image_index += 1
自動將圖片索引遞增，切換到下一張圖片。
if st.session_state.image_index >= len(defect_images_id_list):
如果圖片索引超出當前批次（lot）中的圖片數量：
將 image_index 重置為 0，
更新 lot_index 至下一個批次（模運算確保循環），
根據新的 lot_index 更新 selected_lot_id，
通過 get_lrf_file 和 get_defect_list 重新獲取新的缺陷圖片列表和缺陷類型。
如果用戶點擊了 Update 按鈕，則 update_pressed 為 True，此時不會自動更新 image_index，保持當前圖片，讓用戶能夠繼續編輯。
重置 Update 按鈕標誌

python
複製
st.session_state.update_pressed = False
無論是否自動切換圖片，最後都將 update_pressed 重置為 False，這樣下一次用戶點擊 Update 時能正確識別。
總結
這段代碼的核心邏輯是：

當有新的標注結果提交並發生變化時，同步並保存到數據庫；
接著根據用戶是否點擊 Update 按鈕決定是否自動切換到下一張圖片：
如果用戶未點擊 Update，則自動更新 image_index（並在必要時更新批次信息）；
如果用戶點擊了 Update，則保持當前圖片，讓用戶可以進一步修改；
最後重置 update_pressed 標誌，確保狀態正確。
這樣可以使得用戶在提交後如果希望留在當前圖片上進行進一步更新，系統不會自動跳轉；反之，則自動進入下一張圖片進行標注。


程式碼沒有看到ｓｕｂｍｉｔ變數
推理花了 幾秒
在这份代码中，并没有直接看到一个叫做 “submit” 的变量，因为提交操作是由 Label Studio 组件内部处理的。当用户在组件界面上点击“Submit”按钮后，组件会将最终的标注结果（即 results_raw）返回给 Streamlit 应用，然后代码会通过检查 results_raw 是否为 None 以及是否与上次提交的数据不同（使用 has_results_raw_changed 函数）来判断用户是否完成了提交。

简单来说，提交事件是由 st_labelstudio 组件内部触发的，返回的结果被存储在 results_raw 中，而代码通过这个返回值来决定后续的保存和页面刷新操作。


详细说明
初始化阶段

A：启动 App 时，通过 st.set_page_config 和 session_state 初始化（包括 previous_results_raw、image_index、lot_index 和新增的 update_pressed）。
B：初始化数据库，确保 results_table 存在。
数据与批次选择

C：读取数据目录，获取所有批次（Lot IDs）。
D：在侧边栏选择当前 Lot ID 和 Image ID。
E：调用 get_lrf_file 和 get_defect_list 获取当前批次中的缺陷文件和图片编号列表。
F：侧边栏中还有“Previous Image”、“Next Image”和新增的“Update”按钮，这些按钮会更新 session_state 中的索引或标志。
图像预处理与任务构造

G：调用 get_image_pair_for_studio_input 分别处理 T 和 Rt 图像，获得经过处理的图片数据和元数据。
H：将处理后的图片（Base64 编码）和元数据组合后，调用 task_generator 构造任务数据，其中 data 部分与 config 中的变量（如 $image1、$image2 等）对应；predictions 部分包含预标注数据（自动生成或已有标注）。
调用 Label Studio 组件

I：将 config、interfaces、user 及任务数据传递给 st_labelstudio 组件，在前端显示标注界面。
用户交互与提交

J：用户在前端对图像进行标注，然后点击 “Submit” 按钮。
K：组件返回标注结果 results_raw。
提交结果处理

L：检查 results_raw 是否不为空且与上次提交不同（通过 has_results_raw_changed 判断）。
如果满足条件，则：
M：调用 sync_labels_across_3images 同步不同图片视图之间的标注信息。
N：调用 save_json_to_sqlite 将最新标注保存到数据库。
判断 Update 按钮状态

O：判断 st.session_state.update_pressed 是否为 True。如果用户点击了 Update 按钮，则 update_pressed 为 True，表示用户希望继续编辑当前图片，不自动切换；否则自动切换到下一张图片。
P：如果 update_pressed 为 False，则自动将 image_index 自增，并在必要时更新 lot_index 和重新获取 defect_images_id_list。
Q：重置 update_pressed 为 False（无论哪种情况，都重置标志）。
R：最后调用 st.rerun() 刷新页面，加载新的任务数据或更新后的标注结果。
这种流程确保：

用户提交标注后，若未点击 Update 按钮，系统自动跳转到下一张图片。
若用户点击 Update，则保持当前图片供进一步编辑。
页面刷新后，最新的标注数据已保存到数据库并显示在界面上。
这样就形成了前后端与用户之间的完整交互流程。



