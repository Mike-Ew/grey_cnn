import tkinter as tk
import random


def draw_diagram(canvas):
    canvas.delete("all")

    # Adjusted positions to push the diagram downward
    top_y = 30
    y_center = 400

    x_positions = {
        "input": 100,
        "conv": 300,
        "relu": 500,
        "pool": 700,
        "flatten": 900,
        "fc": 1100,
        "softmax": 1300,
    }

    box_scale = 4.0
    min_box_size = 50
    box_spacing = 10

    font_title = ("Helvetica", 14, "bold")
    font_label = ("Helvetica", 12)
    font_small = ("Helvetica", 10)

    def draw_top_label(x, label):
        canvas.create_text(
            x, top_y, text=label, font=font_title, justify="center", fill="#333"
        )

    def draw_input(x, C, H, W):
        box_w = max(min_box_size, int(W * box_scale * 0.2))
        box_h = max(min_box_size, int(H * box_scale * 0.2))
        top = y_center - box_h / 2
        left = x - box_w / 2
        canvas.create_rectangle(left, top, left + box_w, top + box_h, fill="#A9A9A9")
        canvas.create_text(
            left + box_w / 2, top + box_h / 2, text="Val=...", font=font_label
        )
        draw_top_label(x, f"Input\n{C}x{H}x{W}")
        # return center line reference
        return (x, y_center, box_w, box_h)

    def draw_conv_filters(x, C, H, W):
        box_w = max(min_box_size, int(W * 0.1 * box_scale))
        box_h = max(min_box_size, int(H * 0.1 * box_scale))
        total_height = C * (box_h + box_spacing)
        start_y = y_center - total_height / 2
        for i in range(C):
            y_top = start_y + i * (box_h + box_spacing)
            left = x - box_w / 2
            canvas.create_rectangle(
                left, y_top, left + box_w, y_top + box_h, fill="#87CEEB"
            )
            avg_val = round(random.uniform(0, 1), 2)
            canvas.create_text(
                left + box_w / 2,
                y_top + box_h / 2,
                text=f"Val={avg_val}",
                font=font_label,
            )
            canvas.create_text(
                left + box_w / 2, y_top - 15, text=f"F{i+1}", font=font_small
            )
        draw_top_label(x, f"Conv\n{C}x{H}x{W}")
        return (x, y_center, box_w, total_height)

    def draw_relu(x, C, H, W):
        box_w = max(min_box_size, int(W * 0.1 * box_scale))
        box_h = max(min_box_size, int(H * 0.1 * box_scale))
        left = x - box_w / 2
        top = y_center - box_h / 2
        canvas.create_rectangle(left, top, left + box_w, top + box_h, fill="#ADD8E6")
        canvas.create_text(
            left + box_w / 2, top + box_h / 2, text="Val=...", font=font_label
        )
        draw_top_label(x, f"ReLU\n{C}x{H}x{W}")
        return (x, y_center, box_w, box_h)

    def draw_pool(x, C, H, W):
        box_w = max(min_box_size, int(W * 0.1 * box_scale))
        box_h = max(min_box_size, int(H * 0.1 * box_scale))
        left = x - box_w / 2
        top = y_center - box_h / 2
        canvas.create_rectangle(left, top, left + box_w, top + box_h, fill="#90EE90")
        canvas.create_text(
            left + box_w / 2, top + box_h / 2, text="Val=...", font=font_label
        )
        draw_top_label(x, f"Pool\n{C}x{H}x{W}")
        return (x, y_center, box_w, box_h)

    def draw_flatten_vertical(x, C, H, W):
        # Return positions of each circle for connectivity
        base_radius = 20

        def radius_for_dim(d):
            return base_radius + d * 0.05

        r_c = radius_for_dim(C)
        r_h = radius_for_dim(H)
        r_w = radius_for_dim(W)

        spacing = 30
        total_height = (r_c * 2) + (r_h * 2) + (r_w * 2) + (2 * spacing)
        start_y = y_center - total_height / 2

        cx = x
        # C circle
        cy_c = start_y + r_c
        canvas.create_oval(cx - r_c, cy_c - r_c, cx + r_c, cy_c + r_c, fill="#FFA500")
        canvas.create_text(cx, cy_c, text="C", font=font_label)

        # H circle
        cy_h = cy_c + r_c + spacing + r_h
        canvas.create_oval(cx - r_h, cy_h - r_h, cx + r_h, cy_h + r_h, fill="#FFA500")
        canvas.create_text(cx, cy_h, text="H", font=font_label)

        # W circle
        cy_w = cy_h + r_h + spacing + r_w
        canvas.create_oval(cx - r_w, cy_w - r_w, cx + r_w, cy_w + r_w, fill="#FFA500")
        canvas.create_text(cx, cy_w, text="W", font=font_label)

        features = C * H * W
        draw_top_label(x, f"Flatten\n{C}x{H}x{W}={features}")
        # Return the positions of the circles so we can connect them
        return (
            cx,
            (cy_c + cy_w) / 2,
            max(r_c, r_h, r_w) * 2,
            total_height,
            (cy_c, cy_h, cy_w),
        )

    def draw_fc(x, neurons):
        # Return positions of each neuron for connectivity
        dot_radius = 15
        spacing = 10
        neuron_positions = []
        total_height = neurons * (2 * dot_radius + spacing)
        start_y = y_center - total_height / 2
        for i in range(neurons):
            cy = start_y + i * (2 * dot_radius + spacing) + dot_radius
            canvas.create_oval(
                x - dot_radius,
                cy - dot_radius,
                x + dot_radius,
                cy + dot_radius,
                fill="white",
                outline="black",
            )
            canvas.create_text(x, cy, text=f"N{i+1}", font=font_label)
            neuron_positions.append((x, cy))
        draw_top_label(x, f"FC\n{neurons} neurons")
        return x, y_center, dot_radius * 2, total_height, neuron_positions

    def draw_softmax(x, classes):
        # Return positions of each output neuron for connectivity
        dot_radius = 15
        spacing = 10
        neurons = classes
        total_height = neurons * (2 * dot_radius + spacing)
        start_y = y_center - total_height / 2
        softmax_positions = []
        for i in range(neurons):
            cy = start_y + i * (2 * dot_radius + spacing) + dot_radius
            canvas.create_oval(
                x - dot_radius,
                cy - dot_radius,
                x + dot_radius,
                cy + dot_radius,
                fill="white",
                outline="black",
            )
            canvas.create_text(x, cy, text=f"O{i+1}", font=font_label)
            softmax_positions.append((x, cy))
        draw_top_label(x, f"Output\n(Softmax)\n{classes} classes")
        return x, y_center, dot_radius * 2, total_height, softmax_positions

    # Draw layers
    (ix, iy, iw, ih) = draw_input(x_positions["input"], 1, 28, 28)
    (cx, cy, cw, ch) = draw_conv_filters(x_positions["conv"], 8, 28, 28)
    (rx, ry, rw, rh) = draw_relu(x_positions["relu"], 8, 28, 28)
    (px, py, pw, ph) = draw_pool(x_positions["pool"], 8, 14, 14)
    (fx, fy, fw, fh, flatten_circles_y) = draw_flatten_vertical(
        x_positions["flatten"], 8, 14, 14
    )
    (fcx, fcy, fcw, fch, fc_neurons) = draw_fc(x_positions["fc"], 10)
    (sx, sy, sw, sh, softmax_neurons) = draw_softmax(x_positions["softmax"], 10)

    # Draw arrows between main layer blocks
    arrow_opts = {"arrow": tk.LAST, "width": 2, "fill": "#333"}
    canvas.create_line(ix + iw / 2 + 20, iy, cx - cw / 2 - 20, cy, **arrow_opts)
    canvas.create_line(cx + cw / 2 + 20, cy, rx - rw / 2 - 20, ry, **arrow_opts)
    canvas.create_line(rx + rw / 2 + 20, ry, px - pw / 2 - 20, py, **arrow_opts)
    canvas.create_line(px + pw / 2 + 20, py, fx - fw / 2 - 20, fy, **arrow_opts)

    # Connect Flatten to FC: fully connected (conceptually)
    flatten_circle_positions = [
        (fx, flatten_circles_y[0]),  # C circle
        (fx, flatten_circles_y[1]),  # H circle
        (fx, flatten_circles_y[2]),  # W circle
    ]

    # Draw multiple lines from Flatten circles to each FC neuron
    for fcx_n, fcy_n in fc_neurons:
        for fx_c, fy_c in flatten_circle_positions:
            canvas.create_line(
                fx_c + fw / 2 + 20, fy_c, fcx_n - fcw / 2 - 20, fcy_n, fill="#555"
            )

    # Draw an arrow indicating flow from Flatten block to FC block
    canvas.create_line(fx + fw / 2 + 20, fy, fcx - fcw / 2 - 20, fcy, **arrow_opts)

    # Now fully connect FC neurons to Softmax neurons
    for fcx_n, fcy_n in fc_neurons:
        for sx_n, sy_n in softmax_neurons:
            canvas.create_line(
                fcx_n + fcw / 2 + 20, fcy_n, sx_n - sw / 2 - 20, sy_n, fill="#555"
            )

    # Draw arrow from FC to Softmax to indicate direction of flow
    canvas.create_line(fcx + fcw / 2 + 20, fcy, sx - sw / 2 - 20, sy, **arrow_opts)

    # Draw a simple legend at the bottom:
    legend_x = 100
    legend_y = 700
    canvas.create_text(
        legend_x, legend_y, text="Legend:", font=("Helvetica", 12, "bold"), anchor="nw"
    )
    legend_items = [
        ("#A9A9A9", "Input Image"),
        ("#87CEEB", "Convolution Filters"),
        ("#ADD8E6", "ReLU Activation Maps"),
        ("#90EE90", "Max Pooling"),
        ("#FFA500", "Flattened Dimensions"),
        ("white", "Fully Connected / Output Neurons"),
    ]
    y_offset = 20
    for color, desc in legend_items:
        canvas.create_rectangle(
            legend_x,
            legend_y + y_offset,
            legend_x + 20,
            legend_y + y_offset + 20,
            fill=color,
            outline="black",
        )
        canvas.create_text(
            legend_x + 30,
            legend_y + y_offset + 10,
            text=desc,
            font=font_label,
            anchor="w",
        )
        y_offset += 30


def main():
    root = tk.Tk()
    root.title("Enhanced CNN Diagram with Full Connectivity")

    canvas = tk.Canvas(root, width=1500, height=1000, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)

    draw_diagram(canvas)

    root.mainloop()


if __name__ == "__main__":
    main()
