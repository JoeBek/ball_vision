#include <sl/Camera.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace sl;

void print_help() {
    cout << "ZED Object Detection Example" << endl;
    cout << "Usage: ./zed_object_detection [optional_svo_file]" << endl;
}

int main(int argc, char **argv) {
    
    // Create ZED camera object
    Camera zed;
    
    // Set initialization parameters
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    
    // Handle SVO file input if provided
    if (argc > 1) {
        init_parameters.input.setFromSVOFile(argv[1]);
    }
    
    // Open the camera
    ERROR_CODE returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error opening ZED camera: " << returned_state << endl;
        return EXIT_FAILURE;
    }
    
    // Enable positional tracking (required for object detection)
    PositionalTrackingParameters positional_tracking_parameters;
    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error enabling positional tracking: " << returned_state << endl;
        zed.close();
        return EXIT_FAILURE;
    }
    
    // Enable object detection
    ObjectDetectionParameters obj_det_params;
    obj_det_params.enable_tracking = true;
    obj_det_params.detection_model = OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;
    
    returned_state = zed.enableObjectDetection(obj_det_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error enabling object detection: " << returned_state << endl;
        zed.close();
        return EXIT_FAILURE;
    }
    
    // Configure runtime parameters
    ObjectDetectionRuntimeParameters detection_parameters_rt;
    detection_parameters_rt.detection_confidence_threshold = 50;
    
    // Create objects to store detection results
    Objects objects;
    Mat image;
    
    cout << "Starting object detection... Press Ctrl+C to exit" << endl;
    
    // Main detection loop
    while (true) {
        // Grab new frame
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            
            // Retrieve left image
            zed.retrieveImage(image, VIEW::LEFT);
            
            // Retrieve detected objects
            zed.retrieveObjects(objects, detection_parameters_rt);
            
            // Process detected objects
            if (objects.is_new && objects.object_list.size() > 0) {
                cout << "\n=== Frame " << objects.timestamp.getNanoseconds() << " ===" << endl;
                cout << "Detected " << objects.object_list.size() << " object(s)" << endl;
                
                // Display information for each detected object
                for (auto &obj : objects.object_list) {
                    cout << "Object ID: " << obj.id << endl;
                    cout << "  Label: " << obj.label << endl;
                    cout << "  Confidence: " << obj.confidence << "%" << endl;
                    cout << "  3D Position: [" 
                         << obj.position.x << ", " 
                         << obj.position.y << ", " 
                         << obj.position.z << "] meters" << endl;
                    cout << "  Tracking State: " << obj.tracking_state << endl;
                    
                    // Display 2D bounding box coordinates
                    cout << "  2D Bounding Box:" << endl;
                    for (int i = 0; i < 4; i++) {
                        cout << "    Point " << i << ": [" 
                             << obj.bounding_box_2d[i].x << ", " 
                             << obj.bounding_box_2d[i].y << "]" << endl;
                    }
                    cout << "  ---" << endl;
                }
            }
        }
        
        // Small delay to prevent excessive CPU usage
        this_thread::sleep_for(chrono::milliseconds(30));
    }
    
    // Cleanup
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();
    
    return EXIT_SUCCESS;
}
