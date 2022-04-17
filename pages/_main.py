import dearpygui.dearpygui as dpg
from controllers import MainController


class MainPage():
    def __init__(self, width, height):
        self.control_window_width = width / 3
        self.control_window_height = height 
        self.image_window_width = min(width-self.control_window_width, height)
        self.image_window_height = self.image_window_width
        self.explorer_window_height = height / 2
        self.explorer_window_width = width / 2

        self.controller = MainController()
        
        self.__draw_control_window_elements()
        self.__draw_image_window_elements()
        self.__draw_explorer_window_elements(show=False)
    
    def __draw_control_window_elements(self):
        with dpg.window(label='Control', width=self.control_window_width, height=self.control_window_height, pos=(0, 0), tag='control_window'):
            dpg.add_button(label='Select Image', callback=self.controller.open_explorer)
            dpg.add_button(label='Run Prediction', callback=self.controller.run_test, tag='run_button')
            dpg.add_text(default_value='ResNet-50 prediction: None', tag='resnet_text')
            dpg.add_text(default_value='EfficientNet-B4 prediction: None', tag='efficientnet_text')
            dpg.add_text(default_value='SReT-S prediction: None', tag='sret_text')

    def __draw_image_window_elements(self):
        with dpg.window(label='Image', width=self.image_window_width, height=self.image_window_height, pos=(self.control_window_width, 0), tag='image_window'):
            pass
    
    def __draw_explorer_window_elements(self, show):
        with dpg.file_dialog(height=self.explorer_window_height,
                             width=self.explorer_window_width,
                             directory_selector=False, show=show,
                             callback=self.controller.select_image,
                             tag='select_image_dialog'):
            dpg.add_file_extension(
                'Image Files (*.png *.jpg *.jpeg){.png,.jpg,.jpeg}',
                color=(150, 255, 150, 255)
            )
