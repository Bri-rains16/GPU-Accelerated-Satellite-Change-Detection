#ifndef CHANGE_DETECTOR_H
#define CHANGE_DETECTOR_H

void run_change_detection_cuda(const unsigned char* img1, 
                               const unsigned char* img2, 
                               unsigned char* output, 
                               int width, int height, int channels, 
                               int threshold);

#endif