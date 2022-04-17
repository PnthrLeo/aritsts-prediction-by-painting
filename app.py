import dearpygui.dearpygui as dpg
from pages import MainPage

WIDTH = 1000
HEIGHT = 500

dpg.create_context()

MainPage(width=WIDTH, height=HEIGHT)

dpg.create_viewport(title='Artists Painting Predictor', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
