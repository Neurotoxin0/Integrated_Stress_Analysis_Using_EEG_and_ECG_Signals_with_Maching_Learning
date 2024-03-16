import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AfterSubmissionDialogComponent } from './after-submission-dialog.component';

describe('AfterSubmissionDialogComponent', () => {
  let component: AfterSubmissionDialogComponent;
  let fixture: ComponentFixture<AfterSubmissionDialogComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AfterSubmissionDialogComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AfterSubmissionDialogComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
