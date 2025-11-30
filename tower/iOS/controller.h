//
//  controller.h
//  tower
//
//  Created by Sol Midnight on 10/17/25.
//

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>

@interface ViewController : UIViewController <MTKViewDelegate>
@property (strong, nonatomic) MTKView *metalView;
@property (strong, nonatomic) CADisplayLink *displayLink;
@end
