Dewarping reproductions with chessboard image.

![example](/example.png)

How to use:
1. Pring a grid from grid folder. Don't scale images.
2. Make digital photo of object. For example, old photo print in a photo album with curved pages.
3. Put the image of chessboard on the photo print. Don't change anything in the setup. Make a photo of chessboard grid.
4. Dewarp the image of photo print with the image of grid.

Using:

    python.exe dewarp_chess.py <chessboard_image.tiff> <chessboard-width> <chessboard-height> <warped_image.tiff> <dewarped_image.tif>

Example:

    python.exe dewarp_chess.py 20210114_164614.jpg 39 29 20210114_164614.jpg 20210114_164614_dewarped.tif

See Python source for details.

