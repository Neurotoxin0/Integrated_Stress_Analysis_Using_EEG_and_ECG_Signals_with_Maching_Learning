import { Component, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzLayoutModule } from 'ng-zorro-antd/layout';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { NzImageModule } from 'ng-zorro-antd/image';
import { NzAvatarModule } from 'ng-zorro-antd/avatar';
import { NzCarouselModule } from 'ng-zorro-antd/carousel';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzTypographyModule } from 'ng-zorro-antd/typography';
import { RouterModule } from '@angular/router';
import { provideRouter } from '@angular/router';
import Swiper from 'swiper';

@Component({
  selector: 'app-welcome',
  standalone: true,
  imports: [CommonModule, RouterOutlet, NzIconModule, NzLayoutModule, NzMenuModule, NzImageModule,NzAvatarModule,NzCarouselModule,NzButtonModule,NzTypographyModule,
    RouterModule,],
  templateUrl: './welcome.component.html',
  styleUrls: ['./welcome.component.scss']
})
export class WelcomeComponent implements OnInit, OnDestroy {
  swiper: any;
  autoplayTimer: any;
  constructor() { }

  ngOnInit() {
    this.swiper = new Swiper('.swiper', {
      direction: 'horizontal', 
      loop: true, // 循环模式选项
  
      slidesPerView: "auto", // 不抽搐
      observer: true,
      observeParents: false,
      // autoplay: true,
      autoplay: {
        disableOnInteraction: false,  //触碰后自动轮播也不会停止
        delay: 2500,
      },

      // 如果需要分页器
      pagination: {
        el: '.swiper-pagination',
      },
      grabCursor: true,
      // 如果需要前进后退按钮
      navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
      },
      
      // slidesPerView: 3,
      centeredSlides: true,
      // centeredSlidesBounds: true,
      // loopedSlides: 3,
    });
    this.startAutoplay();
  }

  startAutoplay() {
    clearInterval(this.autoplayTimer);
    this.autoplayTimer = setInterval(() => {
      this.swiper.slideNext();
      //console.log('next');
    }, 2500);
  }

  prevHandler() {
   this.swiper.slidePrev();
  }

  nextHandler() {
    this.swiper.slideNext();
  }

  ngOnDestroy(): void {
      clearInterval(this.autoplayTimer);
  }

}
