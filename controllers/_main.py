import dearpygui.dearpygui as dpg
import models


class MainController():
    def __init__(self):
        artists_df_path = './data/artists.csv'
        ef_net_weights_path = './data/models_weights/efficient_b4_03.pt'
        res_net_weights_path = './data/models_weights/resnet_50_01.pt'
        sret_weights_path = './data/models_weights/sret-s_01.pt'
        
        self.ef_net = models.EfficientNet(ef_net_weights_path, artists_df_path)
        self.res_net = models.ResNet(res_net_weights_path, artists_df_path)
        self.sret = models.SReT(sret_weights_path, artists_df_path)
        
        self.image_path = None
    
    def open_explorer(self, sender, app_data):
        dpg.show_item('select_image_dialog')
        dpg.focus_item('select_image_dialog')
    
    def select_image(self, sender, app_data):
        image_path = app_data['file_path_name']
        self.__show_image(image_path)
        self.image_path = image_path

    def __show_image(self, image_path):
        if dpg.does_item_exist('image'):
            dpg.delete_item('image')
        if dpg.does_item_exist('drawlist'):
            dpg.delete_item('drawlist')
        
        image_width, image_height, _, image_data = dpg.load_image(image_path)
        
        with dpg.texture_registry():
            dpg.add_static_texture(image_width, image_height, image_data, tag='image')

        drawlist_width = dpg.get_item_configuration('image_window')['width']
        drawlist_height = dpg.get_item_configuration('image_window')['height']
        
        if image_height > image_width:
            image_width = image_width * (drawlist_height / image_height)
            image_height = drawlist_height
        else:
            image_height = image_height * (drawlist_width / image_width)
            image_width = drawlist_width
        
        with dpg.drawlist(width=drawlist_width-15, height=drawlist_height-15, tag='drawlist', parent='image_window'):
            dpg.draw_image('image', (0, 0), (image_width, image_height), uv_min=(0, 0), uv_max=(1, 1))
    
    def run_test(self, sender, app_data):
        ef_net_prediction = self.ef_net.predict(self.image_path)
        dpg.set_value('efficientnet_text', f'EfficientNet-B4 prediction: {ef_net_prediction}')
        
        res_net_prediction = self.res_net.predict(self.image_path)
        dpg.set_value('resnet_text', f'ResNet-50 prediction: {res_net_prediction}')
        
        sret_prediction = self.sret.predict(self.image_path)
        dpg.set_value('sret_text', f'SReT-S prediction: {sret_prediction}')
