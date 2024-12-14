import tkinter as tk
from src.layout import DiagramLayout
from src.dynamic import DiagramDynamic
from src.data import load_mnist_data
from src.network import Network
import numpy as np


def main():
    root = tk.Tk()
    root.title("Modular CNN Diagram")

    # Create frame to hold scrollbars and canvas
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)

    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = tk.Canvas(
        frame,
        width=1500,
        height=800,
        bg="white",
        xscrollcommand=hbar.set,
        yscrollcommand=vbar.set,
    )
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    hbar.config(command=canvas.xview)
    vbar.config(command=canvas.yview)

    # Load the actual MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Initialize the network
    net = Network()
    # Run a dummy forward pass to set shapes if needed
    net.forward(X_train[:1])

    layout = DiagramLayout(canvas)
    layout.draw_base_diagram()

    dynamic = DiagramDynamic(canvas, layout)

    current_index = [0]  # We'll cycle through training images

    def simulate_update():
        # Get current image and label
        i = current_index[0]
        img_4d = X_train[i : i + 1]  # shape (1,1,28,28)
        label = y_train[i]

        # Convert to uint8 for display
        img_2d = (img_4d[0, 0] * 255).astype(np.uint8)  # shape (28,28)

        # Update input image
        dynamic.update_input_image(img_2d)

        # Run forward_with_intermediates to get real network outputs
        intermediates = net.forward_with_intermediates(img_4d)

        # Update GUI with real intermediate values and label
        dynamic.update_values(intermediates, label)

        # Move to next image
        current_index[0] = (current_index[0] + 1) % X_train.shape[0]

        # Schedule next update in 2 seconds
        root.after(2000, simulate_update)

    simulate_update()

    canvas.config(scrollregion=canvas.bbox("all"))
    root.mainloop()


if __name__ == "__main__":
    main()
