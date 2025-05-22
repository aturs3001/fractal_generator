
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

# =============================================================================
# fractal_generator.py
# Author: Aric Hurkman
# Description: Fractal generator for Mandelbrot, Julia, Sierpinski, and Burning Ship fractals.
# All Rights Reserved
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

class FractalGenerator:
    def __init__(self, width=800, height=800):
        # Initialize the fractal generator with image width and height
        self.width = width
        self.height = height
    
    def mandelbrot(self, xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0, max_iter=100):
        """Generate the Mandelbrot set"""
        # Create coordinate arrays for the complex plane
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Combine X and Y to form complex numbers
        C = X + 1j * Y
        Z = np.zeros_like(C)
        
        # Array to hold the number of iterations for each point
        iterations = np.zeros(C.shape, dtype=int)
        
        # Iterate to determine divergence
        for i in range(max_iter):
            # Mask for points that have not diverged
            mask = np.abs(Z) <= 2
            # Update Z for non-diverged points
            Z[mask] = Z[mask]**2 + C[mask]
            # Update iteration count
            iterations[mask] = i
        
        return iterations
    
    def julia_set(self, c=-0.7 + 0.27015j, xmin=-2, xmax=2, ymin=-2, ymax=2, max_iter=100):
        """Generate a Julia set with parameter c"""
        # Create coordinate arrays for the complex plane
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Combine X and Y to form complex numbers
        Z = X + 1j * Y
        
        # Array to hold the number of iterations for each point
        iterations = np.zeros(Z.shape, dtype=int)
        
        # Iterate to determine divergence
        for i in range(max_iter):
            # Mask for points that have not diverged
            mask = np.abs(Z) <= 2
            # Update Z for non-diverged points
            Z[mask] = Z[mask]**2 + c
            # Update iteration count
            iterations[mask] = i
        
        return iterations
    
    def sierpinski_triangle(self, iterations=10):
        """Generate Sierpinski triangle using chaos game method"""
        # Define the three vertices of the triangle
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        
        # Number of points to generate
        n_points = 50000
        
        # Starting point (random)
        current_point = np.random.random(2)
        points = []
        
        # Generate points using chaos game
        for _ in range(n_points):
            # Choose random vertex
            vertex = vertices[np.random.randint(0, 3)]
            # Move halfway to chosen vertex
            current_point = (current_point + vertex) / 2
            points.append(current_point.copy())
        
        return np.array(points)
    
    def burning_ship(self, xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0, max_iter=100):
        """Generate the Burning Ship fractal"""
        # Create coordinate arrays for the complex plane
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Combine X and Y to form complex numbers
        C = X + 1j * Y
        Z = np.zeros_like(C)
        
        # Array to hold the number of iterations for each point
        iterations = np.zeros(C.shape, dtype=int)
        
        # Iterate to determine divergence
        for i in range(max_iter):
            # Mask for points that have not diverged
            mask = np.abs(Z) <= 2
            # Burning ship formula: z = (|Re(z)| + i|Im(z)|)^2 + c
            Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag))**2 + C[mask]
            # Update iteration count
            iterations[mask] = i
        
        return iterations
    
    def plot_fractal(self, data, title, colormap='hot', save_path=None):
        """Plot and optionally save fractal image"""
        plt.figure(figsize=(12, 12))
        
        # Plot Sierpinski triangle as scatter, others as images
        if title == "Sierpinski Triangle":
            plt.scatter(data[:, 0], data[:, 1], s=0.1, c='black')
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.0)
            plt.gca().set_aspect('equal')
        else:
            plt.imshow(data, extent=[-2.5, 1.5, -2.0, 2.0], 
                      cmap=colormap, origin='lower', interpolation='bilinear')
            plt.colorbar(label='Iterations to divergence')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Save the image if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fractal saved as {save_path}")
        
        plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate beautiful fractal images')
    parser.add_argument('--type', choices=['mandelbrot', 'julia', 'sierpinski', 'burning_ship', 'all'], 
                       default='all', help='Type of fractal to generate')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=800, help='Image height')
    parser.add_argument('--iterations', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--save', action='store_true', help='Save images to disk')
    parser.add_argument('--output_dir', default='fractals', help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize fractal generator
    generator = FractalGenerator(args.width, args.height)
    
    # Color maps for different fractals
    colormaps = {
        'mandelbrot': 'hot',
        'julia': 'plasma',
        'burning_ship': 'viridis',
        'sierpinski': 'binary'
    }
    
    # Generate and plot Mandelbrot set
    if args.type == 'mandelbrot' or args.type == 'all':
        print("Generating Mandelbrot set...")
        mandelbrot_data = generator.mandelbrot(max_iter=args.iterations)
        save_path = os.path.join(args.output_dir, 'mandelbrot.png') if args.save else None
        generator.plot_fractal(mandelbrot_data, "Mandelbrot Set", 
                             colormaps['mandelbrot'], save_path)
    
    # Generate and plot Julia sets
    if args.type == 'julia' or args.type == 'all':
        print("Generating Julia set...")
        # Try different Julia set parameters
        julia_params = [
            (-0.7 + 0.27015j, "Julia Set (c = -0.7 + 0.27i)"),
            (-0.8 + 0.156j, "Julia Set (c = -0.8 + 0.156i)"),
            (0.3 - 0.5j, "Julia Set (c = 0.3 - 0.5i)")
        ]
        
        for i, (c, title) in enumerate(julia_params):
            julia_data = generator.julia_set(c=c, max_iter=args.iterations)
            save_path = os.path.join(args.output_dir, f'julia_{i+1}.png') if args.save else None
            generator.plot_fractal(julia_data, title, colormaps['julia'], save_path)
    
    # Generate and plot Sierpinski triangle
    if args.type == 'sierpinski' or args.type == 'all':
        print("Generating Sierpinski triangle...")
        sierpinski_data = generator.sierpinski_triangle()
        save_path = os.path.join(args.output_dir, 'sierpinski.png') if args.save else None
        generator.plot_fractal(sierpinski_data, "Sierpinski Triangle", 
                             colormaps['sierpinski'], save_path)
    
    # Generate and plot Burning Ship fractal
    if args.type == 'burning_ship' or args.type == 'all':
        print("Generating Burning Ship fractal...")
        burning_ship_data = generator.burning_ship(max_iter=args.iterations)
        save_path = os.path.join(args.output_dir, 'burning_ship.png') if args.save else None
        generator.plot_fractal(burning_ship_data, "Burning Ship Fractal", 
                             colormaps['burning_ship'], save_path)

def interactive_mode():
    """Interactive mode for exploring fractals with custom parameters"""
    generator = FractalGenerator()
    
    print("\n=== Interactive Fractal Explorer ===")
    print("1. Mandelbrot Set")
    print("2. Julia Set")
    print("3. Custom Julia Set")
    print("4. Sierpinski Triangle")
    print("5. Burning Ship")
    
    # Prompt user for fractal type
    choice = input("\nSelect fractal type (1-5): ").strip()
    
    if choice == '1':
        # Mandelbrot interactive
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.mandelbrot(max_iter=iterations)
        generator.plot_fractal(data, "Mandelbrot Set", 'hot')
    
    elif choice == '2':
        # Julia set interactive (default c)
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.julia_set(max_iter=iterations)
        generator.plot_fractal(data, "Julia Set", 'plasma')
    
    elif choice == '3':
        # Custom Julia set
        real_part = float(input("Enter real part of c (e.g., -0.7): "))
        imag_part = float(input("Enter imaginary part of c (e.g., 0.27): "))
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        c = complex(real_part, imag_part)
        data = generator.julia_set(c=c, max_iter=iterations)
        generator.plot_fractal(data, f"Julia Set (c = {c})", 'plasma')
    
    elif choice == '4':
        # Sierpinski triangle interactive
        data = generator.sierpinski_triangle()
        generator.plot_fractal(data, "Sierpinski Triangle", 'binary')
    
    elif choice == '5':
        # Burning Ship interactive
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.burning_ship(max_iter=iterations)
        generator.plot_fractal(data, "Burning Ship Fractal", 'viridis')

if __name__ == "__main__":
    import sys
    # If no command-line arguments, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        # Otherwise, run main with arguments
        main()

def main():
    parser = argparse.ArgumentParser(description='Generate beautiful fractal images')
    parser.add_argument('--type', choices=['mandelbrot', 'julia', 'sierpinski', 'burning_ship', 'all'], 
                       default='all', help='Type of fractal to generate')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=800, help='Image height')
    parser.add_argument('--iterations', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--save', action='store_true', help='Save images to disk')
    parser.add_argument('--output_dir', default='fractals', help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize fractal generator
    generator = FractalGenerator(args.width, args.height)
    
    # Color maps for different fractals
    colormaps = {
        'mandelbrot': 'hot',
        'julia': 'plasma',
        'burning_ship': 'viridis',
        'sierpinski': 'binary'
    }
    
    if args.type == 'mandelbrot' or args.type == 'all':
        print("Generating Mandelbrot set...")
        mandelbrot_data = generator.mandelbrot(max_iter=args.iterations)
        save_path = os.path.join(args.output_dir, 'mandelbrot.png') if args.save else None
        generator.plot_fractal(mandelbrot_data, "Mandelbrot Set", 
                             colormaps['mandelbrot'], save_path)
    
    if args.type == 'julia' or args.type == 'all':
        print("Generating Julia set...")
        # Try different Julia set parameters
        julia_params = [
            (-0.7 + 0.27015j, "Julia Set (c = -0.7 + 0.27i)"),
            (-0.8 + 0.156j, "Julia Set (c = -0.8 + 0.156i)"),
            (0.3 - 0.5j, "Julia Set (c = 0.3 - 0.5i)")
        ]
        
        for i, (c, title) in enumerate(julia_params):
            julia_data = generator.julia_set(c=c, max_iter=args.iterations)
            save_path = os.path.join(args.output_dir, f'julia_{i+1}.png') if args.save else None
            generator.plot_fractal(julia_data, title, colormaps['julia'], save_path)
    
    if args.type == 'sierpinski' or args.type == 'all':
        print("Generating Sierpinski triangle...")
        sierpinski_data = generator.sierpinski_triangle()
        save_path = os.path.join(args.output_dir, 'sierpinski.png') if args.save else None
        generator.plot_fractal(sierpinski_data, "Sierpinski Triangle", 
                             colormaps['sierpinski'], save_path)
    
    if args.type == 'burning_ship' or args.type == 'all':
        print("Generating Burning Ship fractal...")
        burning_ship_data = generator.burning_ship(max_iter=args.iterations)
        save_path = os.path.join(args.output_dir, 'burning_ship.png') if args.save else None
        generator.plot_fractal(burning_ship_data, "Burning Ship Fractal", 
                             colormaps['burning_ship'], save_path)

def interactive_mode():
    """Interactive mode for exploring fractals with custom parameters"""
    generator = FractalGenerator()
    
    print("\n=== Interactive Fractal Explorer ===")
    print("1. Mandelbrot Set")
    print("2. Julia Set")
    print("3. Custom Julia Set")
    print("4. Sierpinski Triangle")
    print("5. Burning Ship")
    
    choice = input("\nSelect fractal type (1-5): ").strip()
    
    if choice == '1':
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.mandelbrot(max_iter=iterations)
        generator.plot_fractal(data, "Mandelbrot Set", 'hot')
    
    elif choice == '2':
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.julia_set(max_iter=iterations)
        generator.plot_fractal(data, "Julia Set", 'plasma')
    
    elif choice == '3':
        real_part = float(input("Enter real part of c (e.g., -0.7): "))
        imag_part = float(input("Enter imaginary part of c (e.g., 0.27): "))
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        
        c = complex(real_part, imag_part)
        data = generator.julia_set(c=c, max_iter=iterations)
        generator.plot_fractal(data, f"Julia Set (c = {c})", 'plasma')
    
    elif choice == '4':
        data = generator.sierpinski_triangle()
        generator.plot_fractal(data, "Sierpinski Triangle", 'binary')
    
    elif choice == '5':
        iterations = int(input("Enter max iterations (default 100): ") or 100)
        data = generator.burning_ship(max_iter=iterations)
        generator.plot_fractal(data, "Burning Ship Fractal", 'viridis')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        interactive_mode()
    else:
        # Command line arguments provided
        main()
