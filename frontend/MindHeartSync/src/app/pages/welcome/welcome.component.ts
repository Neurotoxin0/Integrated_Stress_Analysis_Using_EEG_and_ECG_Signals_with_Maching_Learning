/*Author: Xilai Wang*/
/*this is the welcome page, a.k.a. home page of the website. */
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
/*this welcome page contains a casousel showcasing multiple use scenario and users of our application and a button to get started. */
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
    //some settings about swiper.
    //this is the latest version of swiper, and it has lots of bugs, or it does not work very well with Angular 17.
    //therefore I've written some extra codes to fix the bugs.
    this.swiper = new Swiper('.swiper', {
      direction: 'horizontal', 
      loop: true, 
  
      slidesPerView: "auto", 
      observer: true,
      observeParents: false,
      // autoplay: true,
      autoplay: {
        disableOnInteraction: false,  
        delay: 2500,
      },

      
      pagination: {
        el: '.swiper-pagination',
      },
      grabCursor: true,
      
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
