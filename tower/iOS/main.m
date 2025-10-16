//
//  main.m
//  tower iOS
//
//  Created by Sol Midnight on 10/9/25.
//

#import <UIKit/UIKit.h>
#import "engine.h"

int main(int argc, char * argv[]) {
    engine_start();
    
    while(true) {
        engine_draw();
    }
}
