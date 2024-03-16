import { Component } from '@angular/core';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzLayoutModule } from 'ng-zorro-antd/layout';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { NzImageModule } from 'ng-zorro-antd/image';
import { NzAvatarModule } from 'ng-zorro-antd/avatar';
import { NzCarouselModule } from 'ng-zorro-antd/carousel';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzTypographyModule } from 'ng-zorro-antd/typography';
import { RouterModule, RouterOutlet } from '@angular/router';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzGridModule } from 'ng-zorro-antd/grid';
import { NzPageHeaderModule } from 'ng-zorro-antd/page-header';
import { NzListModule } from 'ng-zorro-antd/list';
import { NzDividerModule } from 'ng-zorro-antd/divider';
import { NzNotificationService } from 'ng-zorro-antd/notification';

@Component({
  selector: 'app-model-selection',
  standalone: true,
  imports: [ NzLayoutModule,NzCardModule,NzDividerModule,NzGridModule,RouterModule,RouterOutlet,NzPageHeaderModule,NzIconModule,NzListModule],
  template: `
  <nz-page-header
      class="site-page-header"
      routerLink="/menu"
      nzBackIcon="left"
      nzTitle="Choose a Model"
      
    ></nz-page-header>
    <nz-divider></nz-divider>
    
  <nz-content>
    <nz-list nzItemLayout="horizontal">
       
      <nz-list-item-meta
          nzAvatar="../../assets/model_icons/MLP.webp"
          [routerLink]="['/userinput', 'mlp']"
          (click) = "sel_model('mlp')"
        >
          <nz-list-item-meta-title>
            MLP Classifier ( best and most recommended )
          </nz-list-item-meta-title>
        </nz-list-item-meta>


        <nz-divider nzDashed="true"></nz-divider>
        

        <nz-list-item-meta
          nzAvatar="../../assets/model_icons/SVC.webp"
          [routerLink]="['/userinput', 'svc']"
          (click) = "sel_model('svc')"
        >
          <nz-list-item-meta-title>
            Support Vector Classifier
          </nz-list-item-meta-title>
        </nz-list-item-meta>


        <nz-divider nzDashed="true"></nz-divider>


        <nz-list-item-meta
          nzAvatar="../../assets/model_icons/gradientboosting.webp"
          [routerLink]="['/userinput', 'gbc']"
          (click) = "sel_model('gbc')"
        >
          <nz-list-item-meta-title>
            Gradient Boosting Classifier
          </nz-list-item-meta-title>
          </nz-list-item-meta>
          <nz-divider nzDashed="true"></nz-divider>

          
        <nz-list-item-meta
          nzAvatar="../../assets/model_icons/decisiontree.webp"
          [routerLink]="['/userinput', 'dtc']"
          (click) = "sel_model('dtc')"
        >
          <nz-list-item-meta-title>
            Decision Tree Classifier
          </nz-list-item-meta-title>
          </nz-list-item-meta>


        <nz-divider nzDashed="true"></nz-divider>


        <nz-list-item-meta
          nzAvatar="../../assets/model_icons/randomforest2.webp"
          [routerLink]="['/userinput', 'rfc']"
          (click) = "sel_model('rfc')"
        >
          <nz-list-item-meta-title>
            Random Forest Classifier
          </nz-list-item-meta-title>
        </nz-list-item-meta>
        <nz-divider nzDashed="true"></nz-divider>
        
      
    </nz-list>
  </nz-content>
  
  `,
  styleUrl: './model-selection.component.scss'
})
export class ModelSelectionComponent {
  
  model_chosen:string='';
  sel_model(modelname: string){
    this.model_chosen = modelname;
  }
}
