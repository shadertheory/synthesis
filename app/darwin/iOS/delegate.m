//
//  delegate.m
//  tower
//
//  Created by Sol Midnight on 10/17/25.
//

#import "delegate.h"
#import "controller.h"

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application
didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.rootViewController = [[ViewController alloc] init];
    [self.window makeKeyAndVisible];
    
    return YES;
}

@end
