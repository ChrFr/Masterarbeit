// Copyright (C) 2014 by Klaus Jung
// All rights reserved.
// Date: 2014-10-02

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.event.*;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


public class Binarize extends JPanel {
	
	private static final long serialVersionUID = 1L;
	private static final int border = 10;
	private static final int maxWidth = 400;
	private static final int maxHeight = 400;
	private static final File openPath = new File(".");
	private static final String title = "Binarisierung";
	private static final String author = "C.Franke & M.Rother";
	private static final String initalOpen = "tools1.png";
	private static final int minThreshold = 0;
	private static final int maxThreshold = 100;
	private static final int initThreshold = 50;   
	private static JFrame frame;
	private JSlider sizeSlider; // to adjust the ImageSize
	private static final int minZoom = 1;
	private static final int maxZoom = 15;
	
	private ImageView srcView;				// source image view
	private ImageView dstView;				// binarized image view
	private int[] binPixels;				// binarized pixels
	
	private JComboBox<String> methodList;	// the selected binarization method
	private JLabel statusLine;				// to print some status text
	private JSlider slider; 				// to adjust the Threshold
	private JCheckBox checkbox; 				// to active the Outlines
	private int sliderVal = 50;				// initial slider-Value
	private int width;						// the image-width
	private int height;						// the image-height
	private double tempT; 					// a temporary Threshold Value
	private double finalT;					// the final Threshold Value
	private int[] histogram;				// a histogram of the Pixel-Values		


	public Binarize() {
        super(new BorderLayout(border, border));
        
        // load the default image
        File input = new File(initalOpen);
        
        if(!input.canRead()) input = openFile(); // file not found, choose another image
        
        srcView = new ImageView(input);
        srcView.setMaxSize(new Dimension(maxWidth, maxHeight));
        
       
		// create an empty destination image
		dstView = new ImageView(srcView.getImgWidth(), srcView.getImgHeight());
		dstView.setMaxSize(new Dimension(maxWidth, maxHeight));
		
		// load image button
        JButton load = new JButton("Bild öffnen");
        load.addActionListener(new ActionListener() {
        	public void actionPerformed(ActionEvent e) {
        		File input = openFile();
        		if(input != null) {
	        		srcView.loadImage(input);
	        		srcView.setMaxSize(new Dimension(maxWidth, maxHeight));
	                binarizeImage();
        		}
        	}        	
        });
        
		JLabel sizeSliderTxt = new JLabel("Zoom:");

		// Slider to adjust the Zoom-Factor
		sizeSlider = new JSlider(JSlider.HORIZONTAL, minZoom, maxZoom, (int) dstView.getZoom());

		// Turn on labels at major tick marks.
		sizeSlider.setMajorTickSpacing(2);
		sizeSlider.setPaintTicks(true);
		sizeSlider.setPaintLabels(true);

		sizeSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				dstView.setZoom((float) sizeSlider.getValue());
			}
		});
		
        // selector for the binarization method
        JLabel methodText = new JLabel("Methode:");
        String[] methodNames = {"Schwellwert in Prozent", "Iso-Data-Algorithmus"};
        
        methodList = new JComboBox<String>(methodNames);
        methodList.setSelectedIndex(0);		// set initial method
        methodList.addActionListener(new ActionListener() {
        	public void actionPerformed(ActionEvent e) {
                binarizeImage();
        	}
        });
        
        // Slider to adjust the Threshold
        slider = new JSlider(JSlider.HORIZONTAL,minThreshold, maxThreshold, initThreshold );
        
        //Turn on labels at major tick marks.
        slider.setMajorTickSpacing(20);
        slider.setMinorTickSpacing(10);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        
        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
            	// get pixels array
                int srcPixels[] = srcView.getPixels();
            	int dstPixels[] = java.util.Arrays.copyOf(srcPixels, srcPixels.length);
            	sliderVal = slider.getValue();
            	binarize(dstPixels, sliderVal);    
            }
        });
        
        
        // Checkbox for Outlines
        checkbox = new JCheckBox("Outlines");
        checkbox.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent event) {
                checkbox = (JCheckBox) event.getSource();
                if (checkbox.isSelected()) {
                	slider.setEnabled(false);
                    drawOutlines();
                } else {
                	slider.setEnabled(true);
                	binarizeImage();
                }
            }
        });
        

        // some status text
        statusLine = new JLabel(" ");
        
        // arrange all controls
        JPanel controls = new JPanel(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.insets = new Insets(0,border,0,0);
        controls.add(load, c);
        controls.add(methodText, c);
        controls.add(methodList, c);
        controls.add(slider, c);
        controls.add(checkbox, c);
        
        JPanel images = new JPanel(new FlowLayout());
        images.add(srcView);
        images.add(dstView);
        
        add(controls, BorderLayout.NORTH);
        add(images, BorderLayout.CENTER);
        add(statusLine, BorderLayout.SOUTH);
               
        setBorder(BorderFactory.createEmptyBorder(border,border,border,border));
        
        // perform the initial binarization
        binarizeImage();
	}
	
	private File openFile() {
        JFileChooser chooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Images (*.jpg, *.png, *.gif)", "jpg", "png", "gif");
        chooser.setFileFilter(filter);
        chooser.setCurrentDirectory(openPath);
        int ret = chooser.showOpenDialog(this);
        if(ret == JFileChooser.APPROVE_OPTION) {
    		frame.setTitle(title + chooser.getSelectedFile().getName());
        	return chooser.getSelectedFile();
        }
        return null;		
	}
	
	private static void createAndShowGUI() {
		// create and setup the window
		frame = new JFrame(title + " - " + author + " - " + initalOpen);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        JComponent newContentPane = new Binarize();
        newContentPane.setOpaque(true); //content panes must be opaque
        frame.setContentPane(newContentPane);

        // display the window.
        frame.pack();
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        Dimension screenSize = toolkit.getScreenSize();
        frame.setLocation((screenSize.width - frame.getWidth()) / 2, (screenSize.height - frame.getHeight()) / 2);
        frame.setVisible(true);
        
        
	}

	public static void main(String[] args) {
        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
	}
	
	
    protected void binarizeImage() {
    	
        String methodName = (String)methodList.getSelectedItem();
        
        // image dimensions
        width = srcView.getImgWidth();
        height = srcView.getImgHeight();
        
        // get pixels arrays
        int srcPixels[] = srcView.getPixels();
        int dstPixels[] = java.util.Arrays.copyOf(srcPixels, srcPixels.length);
    	    	
    	String message = "Binarisieren mit \"" + methodName + "\"";

    	statusLine.setText(message);

		long startTime = System.currentTimeMillis();
		
    	switch(methodList.getSelectedIndex()) {
    	case 0:	// Threshold in Percent
    		binarize(dstPixels, sliderVal);
    		slider.setEnabled(true);
    		break;
    	case 1:	// Call the ISO-Data-Algorithmus with a Start-Treshold t0 = 127
    		createHistogram(dstPixels);
    		// initialize the temporary Threshold-Value
    		tempT = 0;
    		isoData(dstPixels, 127);

    		slider.setEnabled(false);
    		break;
    	}

		long time = System.currentTimeMillis() - startTime;

        frame.pack();
        
        if(methodName.equals("Iso-Data-Algorithmus"))
        	statusLine.setText(message + " in " + time + " ms. Errechneter Schwellwert: " + (int)finalT + " (=" + Math.round((100.0/255.0)*finalT) + "%)");
    }
    
    

    void binarize(int pixels[], float sliderVal) {
    	
    	float pixelVal = 2.56f*sliderVal;
    	       
    	for(int i = 0; i < pixels.length; i++) {
    		int gray = ((pixels[i] & 0xff) + ((pixels[i] & 0xff00) >> 8) + ((pixels[i] & 0xff0000) >> 16)) / 3;
    		pixels[i] = gray < pixelVal ? 0xff000000 : 0xffffffff;
    	}

    	binPixels = java.util.Arrays.copyOf(pixels, pixels.length);
    	dstView.setPixels(pixels, width, height);
    }
    

    
	private int[] createHistogram(int[] dstPixels){
		ArrayList<Integer> grayValues = new ArrayList<Integer>();
		histogram = new int[256];

		for (int i=0; i<dstPixels.length; i++){
			int gray = ((dstPixels[i] & 0xff) + ((dstPixels[i] & 0xff00) >> 8) + ((dstPixels[i] & 0xff0000) >> 16)) / 3;
			grayValues.add(gray);
		}

		for(int i = 0; i < 256; i++) {
			histogram[i] = Collections.frequency(grayValues, i);
		}
		return histogram;
	}
    
    

    private void isoData(int pixels[], int t0) {

    	int pA = 0;
    	int pA2 = 0;

    	for(int j = 0; j < t0; j++){
    		int pj = histogram[j];
    	    pA += pj;
    	    pA2 += pj*j;
    	}
    	
    	// No Pixels below the Threshold
    	if(pA == 0){
    		 pA = 1;
    		 pA2 = 1;
    	 }

    	double uA = pA2 / pA;;

        int pB = 0;
    	int pB2 = 0;

    	for(int j = t0; j < 256; j++){
    	    pB += histogram[j];
    	    pB2 += histogram[j]*j;
    	}
    	
    	// No Pixels above the Threshold
    	if(pB == 0){
   		 	pB = 1;
   		 	pB2 = 1;
    	}

    	double uB = pB2 / pB;
      
    	// No Pixels below the Threshold
        if(uA == 1){
        	uA = uB;
        }
        
        // No Pixels above the Threshold
        if(uB == 1){
        	uB = uA;
        }
        	
        double t = (uA + uB)/2;
        
        if (tempT == t) {
        	finalT = t;
        	double tP = (100.0/255.0)*t;
        	slider.setValue((int)tP);
        	binarize(pixels, (int)tP);;
        } else {
        	tempT = t;
        	isoData(pixels, (int)t);
        }
    }
    
    
 
    private void drawOutlines(){

    	int[] outPixels = java.util.Arrays.copyOf(binPixels, binPixels.length);
    	int[] eroPixels = erosion();
    	
    	for(int i = 0; i < binPixels.length; i++) {
    		if(binPixels[i] == 0xff000000 && eroPixels[i] == 0xffffffff){
    			outPixels[i] = 0xff000000;
    		}else{
    			outPixels[i] = 0xffffffff;
    		}
    	}
    	dstView.setPixels(outPixels, width, height);
    	
    	// dilation();
    }
    
    
    
    private int[] erosion(){
    	
    	int[] eroPixels = java.util.Arrays.copyOf(binPixels, binPixels.length);
        	
        	for(int i = 0; i < binPixels.length; i++) {
        		
        		eroPixels[i] = 0xffffffff;
    			
        		// Apply Filter just for Black Pixels
        		if(binPixels[i] == 0xff000000){
        			// Check, if left Pixel is Black, inside the Range of the Array and inside the Image
    	    		if(i-1 >= 0 && binPixels[i-1] == 0xff000000 && ((i) % width != 0)){
    	    			// Check, if right Pixel is Black, inside the Range of the Array and inside the Image
    	    			if(i+1 < binPixels.length && binPixels[i+1] == 0xff000000 && ((i+1) % width != 0)){
    	    				// Check, if upper Pixel is Black and inside the Range of the Array
    	    				if(i-width >= 0 && binPixels[i-width] == 0xff000000){
    	    					// Check, if lower Pixel is Black and inside the Range of the Array
    	    					if(i+width < binPixels.length && binPixels[i+width] == 0xff000000){
    	    						// Draw the Pixel
    	    		    			eroPixels[i] = 0xff000000;
    	    		    		} 
    	    	    		} 
    		    		} 
    	    		} 
        		}
         	}
        	return eroPixels;
        }
   
    
  
//    private void dilation(){
//
//    	int[] dilPixels = java.util.Arrays.copyOf(binPixels, binPixels.length);
//
//    	for(int i = 0; i < binPixels.length; i++) {
//			
//    		// Apply Filter just for Black Pixels
//    		if(binPixels[i] == 0xff000000){
//    				
//	    		// Draw Left Pixel Black if it is White, inside the Range of the Array and inside the Image
//	    		if(i-1 >= 0 && binPixels[i-1] == 0xffffffff && ((i) % width != 0)){
//	    			dilPixels[i-1] = 0xff000000;
//	    		} 
//	    		
//	    		// Draw Right Pixel Black if it is White and inside the Range of the Array
//	    		if(i+1 < binPixels.length && binPixels[i+1] == 0xffffffff && ((i+1) % width != 0)){
//	    			dilPixels[i+1] = 0xff000000;
//	    		} 
//	    		
//	    		// Draw Upper Pixel Black if it is White and inside the Range of the Array
//	    		if(i-width >= 0 && binPixels[i-width] == 0xffffffff){
//	    			dilPixels[i-width] = 0xff000000;
//	    		} 
//	    		
//	    		// Draw Lower Pixel Black if it is White and inside the Range of the Array
//	    		if(i+width < binPixels.length && binPixels[i+width] == 0xffffffff){
//	    			dilPixels[i+width] = 0xff000000;
//	    		} 
//    		}
//
//     	}
//    	dstView.setPixels(dilPixels, width, height);
//    }
    
   

}
    
