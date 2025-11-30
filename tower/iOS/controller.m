//
//  controller.m
//  tower
//
//  Created by Sol Midnight on 10/17/25.
//
// ViewController.m
#import "controller.h"
#import "engine.h"
#import <QuartzCore/CADisplayLink.h>


@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    NSLog(@"Engine called!");
    engine_start(self);
    NSLog(@"Engine done!");
    self.displayLink = [CADisplayLink displayLinkWithTarget:self
                                                   selector:@selector(updateFrame)];
    [self.displayLink addToRunLoop:[NSRunLoop mainRunLoop]
                            forMode:NSRunLoopCommonModes];
}


- (void)updateFrame {
    // Called on every display refresh (~60 FPS on standard displays)
    NSLog(@"Frame update");
    engine_draw();
}

- (void)dealloc {
    [self.displayLink invalidate];
    self.displayLink = nil;
}

@end
