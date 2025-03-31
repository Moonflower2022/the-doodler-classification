from quickdraw import QuickDrawDataGroup

def get_and_show_drawings(category, num_drawings):
    data_group = QuickDrawDataGroup(category, recognized=True, max_drawings=num_drawings)
    
    
    for i in range(num_drawings):
        drawing = data_group.get_drawing(i)
        drawing.get_image().show()

if __name__ == "__main__":
    # Example usage: get 5 cat drawings
    get_and_show_drawings('apple', 5)