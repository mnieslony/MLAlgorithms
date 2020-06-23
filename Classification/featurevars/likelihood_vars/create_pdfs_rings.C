void create_pdfs_rings(){

	int n_bins = 500;

	TH1F *pdf_single_charge = new TH1F("pdf_beam_single_charge","pdf_beam_single_charge",n_bins,10,100);
	TH1F *pdf_single_time = new TH1F("pdf_beam_single_time","pdf_beam_single_time",n_bins,0,20);
	TH1F *pdf_single_thetaB = new TH1F("pdf_beam_single_thetaB","pdf_beam_single_thetaB",n_bins,-1.2,2.2);
	TH1F *pdf_single_phiB = new TH1F("pdf_beam_single_phiB","pdf_beam_single_phiB",n_bins,-3.2,3.2);
	TH1F *pdf_multi_charge = new TH1F("pdf_beam_multi_charge","pdf_beam_multi_charge",n_bins,10,100);
	TH1F *pdf_multi_time = new TH1F("pdf_beam_multi_time","pdf_beam_multi_time",n_bins,0,20);
	TH1F *pdf_multi_thetaB = new TH1F("pdf_beam_multi_thetaB","pdf_beam_multi_thetaB",n_bins,-1.2,2.2);
	TH1F *pdf_multi_phiB = new TH1F("pdf_beam_multi_phiB","pdf_beam_multi_phiB",n_bins,-3.2,3.2);


	// Single-/Multi-Ring file is combined in one file

	TFile *file_rings = new TFile("data/beam_DigitThr10_0_4996.root","READ");
	TTree *classification_rings = (TTree*) file_rings->Get("classification_tree");
	int nentries = classification_rings->GetEntries();

	std::vector<double>* event_charge = new std::vector<double>;
	std::vector<double>* event_time = new std::vector<double>;
	std::vector<double>* event_theta = new std::vector<double>;
	std::vector<double>* event_phi = new std::vector<double>;
	bool fMCMultiRing;

	classification_rings->SetBranchAddress("PMTQVector",&event_charge);
	classification_rings->SetBranchAddress("PMTTVector",&event_time);
	classification_rings->SetBranchAddress("PMTThetaBaryVector",&event_theta);
	classification_rings->SetBranchAddress("PMTPhiBaryVector",&event_phi);
	classification_rings->SetBranchAddress("MCMultiRing",&fMCMultiRing);


	for (int i=0; i < nentries; i++){
		classification_rings->GetEntry(i);
		for (int i_q = 0 ; i_q < event_charge->size(); i_q++){
			if (fMCMultiRing){
				pdf_multi_charge->Fill(event_charge->at(i_q));
				pdf_multi_time->Fill(event_time->at(i_q));
				pdf_multi_thetaB->Fill(event_theta->at(i_q));
				pdf_multi_phiB->Fill(event_phi->at(i_q));
			}
			else {
				pdf_single_charge->Fill(event_charge->at(i_q));
				pdf_single_time->Fill(event_time->at(i_q));
				pdf_single_thetaB->Fill(event_theta->at(i_q));
				pdf_single_phiB->Fill(event_phi->at(i_q));
			}
		}
	}	
	
	pdf_single_charge->Sumw2();
	pdf_single_time->Sumw2();
	pdf_single_thetaB->Sumw2();
	pdf_single_phiB->Sumw2();
	pdf_multi_charge->Sumw2();
	pdf_multi_time->Sumw2();
	pdf_multi_thetaB->Sumw2();
	pdf_multi_phiB->Sumw2();

	pdf_single_charge->Scale(1./pdf_single_charge->Integral());
	pdf_single_time->Scale(1./pdf_single_time->Integral());
	pdf_single_thetaB->Scale(1./pdf_single_thetaB->Integral());
	pdf_single_phiB->Scale(1./pdf_single_phiB->Integral());
	pdf_multi_charge->Scale(1./pdf_multi_charge->Integral());
	pdf_multi_time->Scale(1./pdf_multi_time->Integral());
	pdf_multi_thetaB->Scale(1./pdf_multi_thetaB->Integral());
	pdf_multi_phiB->Scale(1./pdf_multi_phiB->Integral());

	// Write pdfs to file

	TFile *out = new TFile("pdf_beam_rings_500bins.root","RECREATE");
	out->cd();
	pdf_single_charge->Write();
	pdf_single_time->Write();
	pdf_single_thetaB->Write();
	pdf_single_phiB->Write();
	pdf_multi_charge->Write();
	pdf_multi_time->Write();
	pdf_multi_thetaB->Write();
	pdf_multi_phiB->Write();
	out->Close();

}
