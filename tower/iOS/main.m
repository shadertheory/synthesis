//
//  main.m
//  tower iOS
//
//  Created by Sol Midnight on 10/9/25.
//

#import <UIKit/UIKit.h>
#import "delegate.h"

int main(int argc, char * argv[]) {
    NSString * appDelegateClassName;
    @autoreleasepool {
        appDelegateClassName = NSStringFromClass([AppDelegate class]);
    }
    UIApplicationMain(argc, argv, nil, appDelegateClassName);
}
