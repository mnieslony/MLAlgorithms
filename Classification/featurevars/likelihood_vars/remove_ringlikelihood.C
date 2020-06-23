void remove_ringlikelihood(){

	TFile *input = new TFile("beam_DigitThr10_0_4996.root");
	TTree *tree = (TTree*) input->Get("classification_tree");
	TFile *output = new TFile("beam_DigitThr10_0_4996_noringlklh.root","RECREATE");
	std::vector<TH1F*> cloned_histograms;
	for (auto&& keyAsObj : *input->GetListOfKeys()){
 		auto key = (TKey*) keyAsObj;
 		cout << key->GetName() << " " << key->GetClassName() << endl;
 		std::string str_th1f = "TH1F";
 		std::string str_qrings = "hist_PMTLikelihoodQRings";
 		std::string str_trings = "hist_PMTLikelihoodTRings";
 		std::string str_thetarings = "hist_PMTLikelihoodThetaRings";
 		std::string str_phirings = "hist_PMTLikelihoodPhiRings";

 		std::cout <<str_th1f.c_str()<<","<<key->GetClassName()<<","<<(key->GetClassName() == str_th1f) << std::endl;
 		if (key->GetClassName() == str_th1f){
 			std::cout <<"key was TH1F"<<std::endl;
 			if (key->GetName() == str_qrings || key->GetName() == str_trings || key->GetName() == str_thetarings || key->GetName() == str_phirings) continue;
 			TH1F *h_clone = (TH1F*) input->Get(key->GetName())->Clone();
 			cloned_histograms.push_back(h_clone);
 		}
	}
	tree->SetBranchStatus("PMTLikelihoodQRings",0);
	tree->SetBranchStatus("PMTLikelihoodTRings",0);
	tree->SetBranchStatus("PMTLikelihoodThetaRings",0);
	tree->SetBranchStatus("PMTLikelihoodPhiRings",0);
	TTree *output_tree = tree->CloneTree(-1,"fast");
	output->cd();
	output_tree->Write();
	for (int i_vec=0; i_vec < cloned_histograms.size(); i_vec++){
		cloned_histograms.at(i_vec)->Write();
	}
	delete input;
	delete output;

}
