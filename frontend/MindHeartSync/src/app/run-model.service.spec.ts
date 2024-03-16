import { TestBed } from '@angular/core/testing';

import { RunModelService } from './run-model.service';

describe('RunModelService', () => {
  let service: RunModelService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(RunModelService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
