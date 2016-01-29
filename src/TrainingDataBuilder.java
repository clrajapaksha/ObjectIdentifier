import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;



public class TrainingDataBuilder {
	
	private String train_path;
	private FeatureDetector detector;
	private DescriptorExtractor extractor;
	private Mat img, descriptors;
	private MatOfKeyPoint keypoints;
	private double[] labels;
	private double[][] data;
	
	public TrainingDataBuilder(String path){
		train_path = path;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//img = Highgui.imread(img_path);
		detector = FeatureDetector.create(FeatureDetector.ORB);
		extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		keypoints = new MatOfKeyPoint();
		descriptors = new Mat();
		labels = new double[(SVMHandler.NumClass)*(SVMHandler.NumImgPerClass)];
		data = new double[SVMHandler.NumClass*SVMHandler.NumImgPerClass][SVMHandler.NumKeyPoints*SVMHandler.DescriptorSize];
	}
	
	public double[][] getTrainingFeatures(){
		for(int i=1; i<=SVMHandler.NumClass; i++){
			for(int j=1; j<=SVMHandler.NumImgPerClass; j++){
				System.out.println(train_path+"/"+i+" ("+j+").jpg");
				img = Highgui.imread(train_path+"/"+i+" ("+j+").jpg",Highgui.CV_LOAD_IMAGE_GRAYSCALE);
				//img.convertTo(img, CvType.CV_32FC1,1.0/255.0);
				detector.detect(img, keypoints);
				extractor.compute(img, keypoints, descriptors);
				//displayImage(Mat2BufferedImage(img));
			    
				for(int row=0;row<SVMHandler.NumKeyPoints;row++){
					for(int col=0; col<SVMHandler.DescriptorSize; col++){
						double pixel = descriptors.get(row, col)[0];
						data[SVMHandler.NumImgPerClass*(i-1)+(j-1)][SVMHandler.NumKeyPoints*row+col] = pixel;
					}
				}
				//data.push_back(img.reshape(1,1));
				labels[(SVMHandler.NumImgPerClass*(i-1))+(j-1)]=i;
			}
		}
		return data;
	}
	
	public double[] getLabels(){
		return labels;
	}
	
	public BufferedImage Mat2BufferedImage(Mat m){
		// source: http://answers.opencv.org/question/10344/opencv-java-load-image-to-gui/
		// Fastest code
		// The output can be assigned either to a BufferedImage or to an Image

		    int type = BufferedImage.TYPE_BYTE_GRAY;
		    if ( m.channels() > 1 ) {
		        type = BufferedImage.TYPE_3BYTE_BGR;
		    }
		    int bufferSize = m.channels()*m.cols()*m.rows();
		    byte [] b = new byte[bufferSize];
		    m.get(0,0,b); // get all the pixels
		    BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
		    final byte[] targetPixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
		    System.arraycopy(b, 0, targetPixels, 0, b.length);  
		    return image;

		}
	 public void displayImage(Image img2)
	 {   
	     //BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
	     ImageIcon icon=new ImageIcon(img2);
	     JFrame frame=new JFrame();
	     frame.setLayout(new FlowLayout());        
	     frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);     
	     JLabel lbl=new JLabel();
	     lbl.setIcon(icon);
	     frame.add(lbl);
	     frame.setVisible(true);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	 }

}
