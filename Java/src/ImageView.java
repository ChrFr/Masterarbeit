// Copyright (C) 2010 by Klaus Jung
// All rights reserved.
// Date: 2010-03-15

import java.awt.BasicStroke;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.SystemColor;
import java.awt.image.BufferedImage;
import java.awt.Font;
import java.awt.Color;
import java.awt.Graphics2D;
import java.io.File;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.geom.AffineTransform;

public class ImageView extends JScrollPane {

	private static final long serialVersionUID = 1L;

	private ImageScreen screen = null;
	private Dimension maxSize = null;
	private int borderX = -1;
	private int borderY = -1;
	private float zoom = 8.0f;

	private double maxViewMagnification = 0.0; // use 0.0 to disable limits
	private boolean keepAspectRatio = true;
	private boolean centered = true;
	private boolean renderContours = false;
	private boolean renderPolygons = false;
	private boolean renderBezierCurves = false;
	private boolean renderImage = true;

	public void setRenderImage(boolean renderImage) {
		this.renderImage = renderImage;
	}
	
	public void setRenderContours(boolean renderContours) {
		this.renderContours = renderContours;
	}

	public void setRenderPolygons(boolean renderPolygons) {
		this.renderPolygons = renderPolygons;
	}

	public void setRenderBezierCurves(boolean renderBezierCurves) {
		this.renderBezierCurves = renderBezierCurves;
	}
	
	public boolean getRenderBezierCurves() {
		return renderBezierCurves;
	}

	int pixels[] = null; // pixel array in ARGB format

	private static ArrayList<ArrayList<Line2D>> contours = new ArrayList<ArrayList<Line2D>>();
	
	
	private static ArrayList<Boolean> isOuterContour = new ArrayList<>();

	private static ArrayList<ArrayList<Line2D>> polygons = new ArrayList<ArrayList<Line2D>>();

	private static ArrayList<Path2D> bezierCurves = new ArrayList<Path2D>();	

	public static ArrayList<Point2D> testPoints = new ArrayList<Point2D>();

	public ImageView(int width, int height) {
		// construct empty image of given size
		BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		init(bi, true);
	}

	public float getZoom() {
		return zoom;
	}

	public void setZoom(float zoom) {
		this.zoom = zoom;

		screen.revalidate();
	}

	// public ArrayList<Line2D> getlineList() {
	// return lineList;
	// }

	public ImageView(File file) {
		// construct image from file
		loadImage(file);
	}

	public void setMaxSize(Dimension dim) {
		// limit the size of the image view
		maxSize = new Dimension(dim);

		Dimension size = new Dimension(maxSize);
		if (size.width - borderX > screen.image.getWidth())
			size.width = screen.image.getWidth() + borderX;
		if (size.height - borderY > screen.image.getHeight())
			size.height = screen.image.getHeight() + borderY;
		setPreferredSize(size);
	}

	public int getImgWidth() {
		return screen.image.getWidth();
	}

	public int getImgHeight() {
		return screen.image.getHeight();
	}

	public void resetToSize(int width, int height) {
		// resize image and erase all content
		if (width == getImgWidth() && height == getImgHeight())
			return;

		screen.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		pixels = new int[getImgWidth() * getImgHeight()];
		screen.image.getRGB(0, 0, getImgWidth(), getImgHeight(), pixels, 0, getImgWidth());

		Dimension size = new Dimension(maxSize);
		if (size.width - borderX > width)
			size.width = width + borderX;
		if (size.height - borderY > height)
			size.height = height + borderY;
		setPreferredSize(size);

		screen.invalidate();
		screen.repaint();
	}

	public int[] getPixels() {
		// get reference to internal pixels array
		if (pixels == null) {
			pixels = new int[getImgWidth() * getImgHeight()];
			screen.image.getRGB(0, 0, getImgWidth(), getImgHeight(), pixels, 0, getImgWidth());
		}
		return pixels;
	}

	public void applyChanges() {
		// if the pixels array obtained by getPixels() has been modified,
		// call this method to make your changes visible
		if (pixels != null)
			setPixels(pixels);
	}

	public void setPixels(int[] pix) {
		// set pixels with same dimension
		setPixels(pix, getImgWidth(), getImgHeight());
	}

	public void setPixels(int[] pix, int width, int height) {
		// set pixels with arbitrary dimension
		if (pix == null || pix.length != width * height)
			throw new IndexOutOfBoundsException();

		if (width != getImgWidth() || height != getImgHeight()) {
			// image dimension changed
			screen.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			pixels = null;
		}

		screen.image.setRGB(0, 0, width, height, pix, 0, width);

		if (pixels != null && pix != pixels) {
			// update internal pixels array
			System.arraycopy(pix, 0, pixels, 0, Math.min(pix.length, pixels.length));
		}

		Dimension size = new Dimension(maxSize);
		if (size.width - borderX > width)
			size.width = width + borderX;
		if (size.height - borderY > height)
			size.height = height + borderY;
		setPreferredSize(size);

		screen.invalidate();
		screen.repaint();
	}

	public double getMaxViewMagnification() {
		return maxViewMagnification;
	}

	// set 0.0 to disable limits
	//
	public void setMaxViewMagnification(double mag) {
		maxViewMagnification = mag;
	}

	public boolean getKeepAspectRatio() {
		return keepAspectRatio;
	}

	public void setKeepAspectRatio(boolean keep) {
		keepAspectRatio = keep;
	}

	public void setCentered(boolean centered) {
		this.centered = centered;
	}

	public void printText(int x, int y, String text) {
		Graphics2D g = screen.image.createGraphics();

		Font font = new Font("TimesRoman", Font.BOLD, 12);
		g.setFont(font);
		g.setPaint(Color.black);
		g.drawString(text, x, y);
		g.dispose();

		updatePixels(); // update the internal pixels array
	}

	public void clearImage() {
		Graphics2D g = screen.image.createGraphics();

		g.setColor(Color.white);
		g.fillRect(0, 0, getImgWidth(), getImgHeight());
		g.dispose();

		updatePixels(); // update the internal pixels array
	}

	public void loadImage(File file) {
		// load image from file
		BufferedImage bi = null;
		boolean success = false;

		try {
			bi = ImageIO.read(file);
			success = true;
		} catch (Exception e) {
			JOptionPane.showMessageDialog(this, "Bild konnte nicht geladen werden.", "Fehler",
					JOptionPane.ERROR_MESSAGE);
			bi = new BufferedImage(200, 150, BufferedImage.TYPE_INT_RGB);
		}

		init(bi, !success);

		if (!success)
			printText(5, getImgHeight() / 2, "Bild konnte nicht geladen werden.");
	}

	public void saveImage(String fileName) {
		try {
			File file = new File(fileName);
			String ext = (fileName.lastIndexOf(".") == -1) ? ""
					: fileName.substring(fileName.lastIndexOf(".") + 1, fileName.length());
			if (!ImageIO.write(screen.image, ext, file))
				throw new Exception("Image save failed");
		} catch (Exception e) {
			JOptionPane.showMessageDialog(this, "Bild konnte nicht geschrieben werden.", "Fehler",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	private void init(BufferedImage bi, boolean clear) {
		screen = new ImageScreen(bi);
		setViewportView(screen);

		maxSize = new Dimension(getPreferredSize());

		if (borderX < 0)
			borderX = maxSize.width - bi.getWidth();
		if (borderY < 0)
			borderY = maxSize.height - bi.getHeight();

		if (clear)
			clearImage();

		pixels = null;
	}

	private void updatePixels() {
		if (pixels != null)
			screen.image.getRGB(0, 0, getImgWidth(), getImgHeight(), pixels, 0, getImgWidth());
	}

	public void addContour(ArrayList<Line2D> contour, boolean isOuter) {

		contours.add(contour);
		isOuterContour.add(isOuter);
	}

	public void clearContour() {
		contours.clear();
	}

	public void clearOuterContourList() {
		isOuterContour.clear();
	}

	public void clearPolygons() {
		polygons.clear();
	}

	public void clearBezierCurves() {
		bezierCurves.clear();
	}

	public ArrayList<Path2D> getBezierCurves() {
		return bezierCurves;
	}

	public ArrayList<ArrayList<Line2D>> getContours() {
		return contours;
	}

	public ArrayList<ArrayList<Line2D>> getPolygons() {
		return polygons;
	}

	public void addPolygon(ArrayList<Line2D> polygon) {
		polygons.add(polygon);
	}

	public void addBezierCurve(Path2D bezierCurve) {
		bezierCurves.add(bezierCurve);
	}

	public void draw() {

		screen.revalidate();
	}

	class ImageScreen extends JComponent {

		private static final long serialVersionUID = 1L;

		private BufferedImage image = null;

		public ImageScreen(BufferedImage bi) {
			super();
			image = bi;
		}

		public void paintComponent(Graphics g) {

			if (image != null) {
				Rectangle r = this.getBounds();

				// limit image view magnification
				if (maxViewMagnification > 0.0) {
					int maxWidth = (int) (image.getWidth() * maxViewMagnification + 0.5);
					int maxHeight = (int) (image.getHeight() * maxViewMagnification + 0.5);

					if (r.width > maxWidth)
						r.width = maxWidth;
					if (r.height > maxHeight)
						r.height = maxHeight;
				}

				// keep aspect ratio
				if (keepAspectRatio) {
					double ratioX = (double) r.width / image.getWidth();
					double ratioY = (double) r.height / image.getHeight();
					if (ratioX < ratioY)
						r.height = (int) (ratioX * image.getHeight() + 0.5);
					else
						r.width = (int) (ratioY * image.getWidth() + 0.5);
				}

				int offsetX = 0;
				int offsetY = 0;

				// set background for regions not covered by image
				if (r.height < getBounds().height) {
					g.setColor(SystemColor.window);
					if (centered)
						offsetY = (getBounds().height - r.height) / 2;
					g.fillRect(0, 0, getBounds().width, offsetY);
					g.fillRect(0, r.height + offsetY, getBounds().width, getBounds().height - r.height - offsetY);
				}

				if (r.width < getBounds().width) {
					g.setColor(SystemColor.window);
					if (centered)
						offsetX = (getBounds().width - r.width) / 2;
					g.fillRect(0, offsetY, offsetX, r.height);
					g.fillRect(r.width + offsetX, offsetY, getBounds().width - r.width - offsetX, r.height);
				}

				if(renderImage)
				// draw image
					g.drawImage(image, offsetX, offsetY, r.width, r.height, this);

				Graphics2D g2D = (Graphics2D) g;

				// draw grid if zoom > 6
				if (zoom >= 6.0f) {

					//drawGrid(g2D, r.width, r.height);
				}
				
				if(renderContours)
					drawContour(g2D, offsetX, offsetY);
				if(renderPolygons)
					drawPolygon(g2D, offsetX, offsetY);
				if(renderBezierCurves)
					drawBezierCurves(g2D, offsetX, offsetY);
				drawTestPoints(g2D, offsetX, offsetY);
			}
		}

		public Dimension getPreferredSize() {
			if (image != null)
				return new Dimension((int) zoom * image.getWidth(), (int) zoom * image.getHeight());
			else
				return new Dimension(100, 60);
		}

		// Draws a Grid in the Image
		public void drawGrid(Graphics2D g2D, int width, int height) {

			g2D.setColor(Color.BLACK);
			g2D.setStroke(new BasicStroke(1.0f));

			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {

					if (x == 0) {
						g2D.drawLine((int) zoom * x, (int) zoom * y, (int) zoom * x + width * (int) zoom,
								(int) zoom * y);
					}

					if (y == 0) {
						g2D.drawLine((int) zoom * x, (int) zoom * y, (int) zoom * x,
								(int) zoom * y + height * (int) zoom);
					}
				}
			}
		}

		// Draws the contour in the Image
		public void drawContour(Graphics2D g2D, int offsetX, int offsetY) {

			g2D.setStroke(new BasicStroke(2.0f));
			int i = 0;

			for (ArrayList<Line2D> contour : contours) {
				if(isOuterContour.get(i)){
					g2D.setColor(Color.RED);
				}else{
					g2D.setColor(Color.ORANGE);
				}
				
				i++;
				for (Line2D line : contour) {

					double x1 = offsetX + line.getX1();
					double y1 = offsetY + line.getY1();
					double x2 = offsetX + line.getX2();
					double y2 = offsetY + line.getY2();

					Line2D zoomedLine = new Line2D.Double(x1 * zoom, y1 * zoom, x2 * zoom, y2 * zoom);

					g2D.draw(zoomedLine);
				}
			}
		}

		// Draws the polygons in the Image
		public void drawPolygon(Graphics2D g2D, int offsetX, int offsetY) {

			g2D.setStroke(new BasicStroke(2.0f));
			g2D.setColor(Color.BLUE);

			for (ArrayList<Line2D> polygon : polygons) {

				for (Line2D line : polygon) {

					double x1 = offsetX + line.getX1();
					double y1 = offsetY + line.getY1();
					double x2 = offsetX + line.getX2();
					double y2 = offsetY + line.getY2();

					Line2D zoomedLine = new Line2D.Double(x1 * zoom, y1 * zoom, x2 * zoom, y2 * zoom);
					double radius = 3;
					g2D.drawOval((int) (x1 * zoom - radius), (int) (y1 * zoom - radius), (int) (2 * radius),
							(int) (2 * radius));

					g2D.draw(zoomedLine);
				}
			}
		}

		// Draws the bezier-curves in the Image
		public void drawBezierCurves(Graphics2D g2D, int offsetX, int offsetY) {
			g2D.setStroke(new BasicStroke(2.0f));
			
			int i = 0;
			for (Path2D path : bezierCurves) {
				
				if(isOuterContour.get(i)){
					g2D.setColor(Color.BLACK);
				} else {
					g2D.setColor(Color.WHITE);
				}
				
				AffineTransform affineTransform = new AffineTransform();
				affineTransform.scale(zoom, zoom);
				Path2D scaledPath = (Path2D) path.clone();
				scaledPath.transform(affineTransform);
				g2D.fill(scaledPath);
				i++;
			}
		}
		
		// Draws the bezier-curves in the Image
		public void drawTestPoints(Graphics2D g2D, int offsetX, int offsetY) {

			g2D.setStroke(new BasicStroke(2.0f));
			g2D.setColor(Color.GREEN);

			for (Point2D point : testPoints) {
				AffineTransform affineTransform = new AffineTransform();
				affineTransform.scale(zoom, zoom);
				double radius = 3;
				g2D.drawOval((int) (point.getX() * zoom - radius), (int) (point.getY() * zoom - radius), (int) (2 * radius),
						(int) (2 * radius));
			}
		}

	}

}