void convert_canvas_to_pdf(){

	TFile *input = new TFile("ClassificationVars_Comparison_beamlike_beam_muon_SingleRing.root");
	TTree *tree = (TTree*) input->Get("classification_tree");
	

	int i_canvas = 0;
	int number_canvas = input->GetListOfKeys()->GetSize();

	for (auto&& keyAsObj : *input->GetListOfKeys()){

 		auto key = (TKey*) keyAsObj;
 		cout << key->GetName() << " " << key->GetClassName() << endl;

 		std::string str_canvas = "TCanvas";
 		
 		if (key->GetClassName() == str_canvas){
 			std::cout <<"key was TCanvas"<<std::endl;
 			TCanvas *c_clone = (TCanvas*) input->Get(key->GetName())->Clone();
 			if (i_canvas == 0) c_clone->Print("ClassificationVars_Comparison_beamlike_beam_muon_SingleRing.pdf(","pdf");
 			else if (i_canvas == number_canvas - 1) c_clone->Print("ClassificationVars_Comparison_beamlike_beam_muon_SingleRing.pdf)","pdf");
 			else c_clone->Print("ClassificationVars_Comparison_beamlike_beam_muon_SingleRing.pdf","pdf");
 			i_canvas++;
 		}
	}

	delete input;

}
