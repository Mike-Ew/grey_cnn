from PIL import Image, ImageTk
import numpy as np


class DiagramLayout:
    def __init__(self, canvas):
        self.canvas = canvas
        self.elements = {}
        self.x_positions = {
            "input": 100,
            "conv": 300,
            "relu_pool": 500,
            "flatten": 700,
            "fc": 900,
            "softmax": 1300,
        }
        self.y_center = 300
        self.top_y = 50

    def draw_base_diagram(self):
        # Input Image
        input_array = np.random.rand(28, 28) * 255
        input_array = input_array.astype(np.uint8)
        pil_img = Image.fromarray(input_array, mode="L")
        pil_img = pil_img.resize((280, 280), Image.NEAREST)
        self.input_photo = ImageTk.PhotoImage(pil_img)

        x_input = self.x_positions["input"]
        image_id = self.canvas.create_image(
            x_input, self.y_center, image=self.input_photo
        )
        self.elements[("input", "image")] = image_id

        # "Val=..." below input
        val_y = self.y_center + 280 / 2 + 20
        val_text_id = self.canvas.create_text(
            x_input, val_y, text="Val=...", font=("Helvetica", 12), fill="black"
        )
        self.elements[("input", "text")] = val_text_id

        # Input label
        self.canvas.create_text(
            x_input,
            self.top_y,
            text="Input\n1x28x28",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # Conv layer visualization (8 filters)
        x_conv = self.x_positions["conv"]
        self.canvas.create_text(
            x_conv,
            self.top_y,
            text="Conv\n(8 filters)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # Create placeholders for 8 filter output images
        filter_count = 8
        filter_img_size = 140  # chosen arbitrarily
        filter_spacing = 20
        total_height = filter_count * (filter_img_size + filter_spacing)
        start_y = self.y_center - total_height / 2

        blank_array = np.zeros((filter_img_size, filter_img_size), dtype=np.uint8)
        blank_pil = Image.fromarray(blank_array, mode="L")
        blank_img = ImageTk.PhotoImage(blank_pil)

        for i in range(filter_count):
            y_top = (
                start_y + i * (filter_img_size + filter_spacing) + filter_img_size / 2
            )
            img_id = self.canvas.create_image(
                self.x_positions["conv"], y_top, image=blank_img
            )
            self.elements[("conv", f"filter_{i}")] = img_id

        # Placeholder box for ReLU/Pool
        x_relu_pool = self.x_positions["relu_pool"]
        box_w, box_h = 100, 100
        left = x_relu_pool - box_w / 2
        top = self.y_center - box_h / 2
        self.canvas.create_rectangle(
            left, top, left + box_w, top + box_h, fill="#C0C0C0"
        )
        self.canvas.create_text(
            x_relu_pool,
            self.top_y,
            text="ReLU/Pool",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # Placeholder box for Flatten
        x_flatten = self.x_positions["flatten"]
        left = x_flatten - box_w / 2
        top = self.y_center - box_h / 2
        self.canvas.create_rectangle(
            left, top, left + box_w, top + box_h, fill="#C0C0C0"
        )
        self.canvas.create_text(
            x_flatten,
            self.top_y,
            text="Flatten",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # FC Layer: 10 neurons
        x_fc = self.x_positions["fc"]
        fc_neurons = 10
        dot_radius = 15
        spacing = 10
        total_height = fc_neurons * (2 * dot_radius + spacing)
        start_y = self.y_center - total_height / 2

        self.canvas.create_text(
            x_fc,
            self.top_y,
            text="FC\n(10 neurons)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        for i in range(fc_neurons):
            cy = start_y + i * (2 * dot_radius + spacing) + dot_radius
            n_id = self.canvas.create_oval(
                x_fc - dot_radius,
                cy - dot_radius,
                x_fc + dot_radius,
                cy + dot_radius,
                fill="white",
                outline="black",
                width=1,
            )
            self.elements[("fc", f"neuron_{i}")] = n_id

        # Softmax predicted class and actual label
        x_softmax = self.x_positions["softmax"]
        softmax_text_y = self.y_center + 200
        class_text_id = self.canvas.create_text(
            x_softmax,
            softmax_text_y,
            text="Predicted: ...",
            font=("Helvetica", 12),
            fill="black",
        )
        self.elements[("softmax", "class_text")] = class_text_id

        label_text_id = self.canvas.create_text(
            x_softmax,
            softmax_text_y + 30,
            text="Label: ...",
            font=("Helvetica", 12),
            fill="black",
        )
        self.elements[("softmax", "label_text")] = label_text_id

        self.canvas.create_text(
            x_softmax,
            self.top_y,
            text="Output\n(Softmax)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # Add placeholders for top-3 classes
        topk_start_y = softmax_text_y + 60
        for i in range(3):
            t_id = self.canvas.create_text(
                x_softmax,
                topk_start_y + i * 20,
                text=f"Top {i+1}: ...",
                font=("Helvetica", 12),
                fill="black",
            )
            self.elements[("softmax", f"topk_class_{i}")] = t_id
