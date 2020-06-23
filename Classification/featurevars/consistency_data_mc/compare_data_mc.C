void compare_data_mc(const char* datafile, const char* mcfile, const char* outfile, const char* configfile, const char* configtypefile, const char* label1="data", const char* label2="MC"){


	std::cout <<"compare_rings: Reading in file "<<datafile<<std::endl;
	bool execute_program = true;


	ifstream file_data(datafile);
	if (!file_data.good()) {
		std::cout <<"Specified data file "<<datafile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}

	ifstream file_mc(mcfile);
	if (!file_mc.good()) {
		std::cout <<"Specified MC file "<<mcfile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}

	std::cout <<"Reading in variable configuration file "<<configfile<<std::endl;
	ifstream file_config(configfile);
	if (!file_config.good()) {
		std::cout <<"Specified variable configuration file "<<configfile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}


	std::cout <<"Reading in variable configuration type file "<<configtypefile<<std::endl;
	ifstream file_configtype(configtypefile);
	if (!file_configtype.good()) {
		std::cout << "Specified variable configuration type file "<<configtypefile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}


	if (execute_program){

		TFile *f_data = new TFile(datafile,"READ");
		TFile *f_mc = new TFile(mcfile,"READ");
		TFile *f_out = new TFile(outfile,"RECREATE");

		std::vector<std::string> vector_varnames;
		std::string temp_string;
		while (!file_config.eof()){
			file_config >> temp_string;
			if (file_config.eof()) break;
			vector_varnames.push_back(temp_string);
		}
		file_config.close();
		
		std::map<std::string,int> map_vartypes;		
		std::map<std::string,int> map_nbins;
		std::map<std::string,double> map_minbin;
		std::map<std::string,double> map_maxbin;

		std::string temp_varname;
		int temp_nbins;
		double temp_minbin;
		double temp_maxbin;
		int temp_type;

		while (!file_configtype.eof()){
			file_configtype >> temp_varname >> temp_type >> temp_nbins >> temp_minbin >> temp_maxbin;
			if (file_configtype.eof()) break;
			map_vartypes.emplace(temp_varname,temp_type);
			map_nbins.emplace(temp_varname,temp_nbins);
			map_minbin.emplace(temp_varname,temp_minbin);
			map_maxbin.emplace(temp_varname,temp_maxbin);
		}
		file_configtype.close();


		TTree *classification_tree = (TTree*) f_data->Get("classification_tree");
		TTree *classification_tree_mc = (TTree*) f_mc->Get("classification_tree");
		int n_entries_data = classification_tree->GetEntries();
		int n_entries_mc = classification_tree_mc->GetEntries();

		std::map<std::string,int> variable_int;
		std::map<std::string,double> variable_double;
		std::map<std::string,bool> variable_bool;
		std::map<std::string,std::vector<double>*> variable_vector;

		std::map<std::string,int> variable_int_mc;
		std::map<std::string,double> variable_double_mc;
		std::map<std::string,bool> variable_bool_mc;
		std::map<std::string,std::vector<double>*> variable_vector_mc;

		for (int i=0; i<int(vector_varnames.size()); i++){

			std::string varname = vector_varnames.at(i);
			int vartype = map_vartypes[varname];
			if (vartype == 1) {
				variable_int.emplace(varname,-1);
				variable_int_mc.emplace(varname,-1);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_int.at(varname));
				classification_tree_mc->SetBranchAddress(varname.c_str(),&variable_int_mc.at(varname));
			}
			else if (vartype ==2) {
				variable_double.emplace(varname,-1);
				variable_double_mc.emplace(varname,-1);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_double.at(varname));
				classification_tree_mc->SetBranchAddress(varname.c_str(),&variable_double_mc.at(varname));
			}
			else if (vartype == 3){
				variable_bool.emplace(varname,0);
				variable_bool_mc.emplace(varname,0);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_bool.at(varname));
				classification_tree_mc->SetBranchAddress(varname.c_str(),&variable_bool_mc.at(varname));
			}
			else if (vartype == 4){
				std::vector<double> *temp_vector = new std::vector<double>;
				std::vector<double> *temp_vector_mc = new std::vector<double>;
				variable_vector.emplace(varname.c_str(),temp_vector);
				variable_vector_mc.emplace(varname.c_str(),temp_vector_mc);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_vector.at(varname));
				classification_tree_mc->SetBranchAddress(varname.c_str(),&variable_vector_mc.at(varname));
			}

		}

		//std::vector<double> *t_corrected = new std::vector<double>;
		//classification_tree->Branch("t_corrected",&t_corrected);

		TH1F *hist_temp_data = nullptr;
		TH1F *hist_temp_mc = nullptr;

		std::vector<TCanvas*> vector_canvas;
        	TLegend *leg = new TLegend(0.75,0.75,0.9,0.9);
        	std::map<std::string,TH1F*> histogram_map_data;
        	std::map<std::string,TH1F*> histogram_map_mc;

		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){

			std::string varname = vector_varnames.at(i_var);
			std::stringstream ss_histname_data, ss_histname_mc;
			ss_histname_data << "hist_"<<vector_varnames.at(i_var) <<"_data";
			ss_histname_mc << "hist_"<<vector_varnames.at(i_var) <<"_mc";
			TH1F *hist_temp_data = new TH1F(ss_histname_data.str().c_str(),ss_histname_data.str().c_str(),map_nbins[varname],map_minbin[varname],map_maxbin[varname]);
			TH1F *hist_temp_mc = new TH1F(ss_histname_mc.str().c_str(),ss_histname_mc.str().c_str(),map_nbins[varname],map_minbin[varname],map_maxbin[varname]);

			histogram_map_data.emplace(vector_varnames.at(i_var),hist_temp_data);
			histogram_map_mc.emplace(vector_varnames.at(i_var),hist_temp_mc);

		}

		std::cout <<"n_entries_data: "<<n_entries_data<<std::endl;

		for (int i_entry=0; i_entry < n_entries_data; i_entry++) {

			std::cout <<"Data entry: "<<i_entry<<", ";
			classification_tree->GetEntry(i_entry);
			//t_corrected->clear();
			for (int i_var=0; i_var < int(vector_varnames.size()); i_var++){
				std::string varname = vector_varnames.at(i_var);
				int vartype = map_vartypes[varname];
				if (vartype == 1){
					int variable = variable_int[varname];
					histogram_map_data[varname]->Fill(variable);
				}
				else if (vartype == 2){
					double variable = variable_double[varname];
					histogram_map_data[varname]->Fill(variable);
				}
				else if (vartype == 3){
					bool variable = variable_bool[varname];
					histogram_map_data[varname]->Fill(variable);
				}
				else if (vartype == 4){
					std::vector<double>* variable = variable_vector[varname];
					std::vector<double>* q_variable = variable_vector["PMTQVector"];
					double min_time=0;
					if (varname == "PMTTVector") min_time = *std::min_element(variable->begin(),variable->end());
					for (int j=0; j < int(variable->size()); j++){
						double single_variable = variable->at(j);
						double q = q_variable->at(j);
						if (q*1.375-20 < 10) continue;
						if (varname == "PMTQVector") {
							single_variable*=1.375;
							single_variable-=20;
						}
						if (varname == "PMTTVector") {
							single_variable-=min_time;
							//std::cout <<"Data time: "<<single_variable<<std::endl;
							//t_corrected->push_back(single_variable);
						}
						histogram_map_data[varname]->Fill(single_variable);
					}
				}
			}
			//classification_tree->Fill();
		}

		for (int i_entry=0; i_entry < n_entries_mc; i_entry++) {

                        classification_tree_mc->GetEntry(i_entry);
                        for (int i_var=0; i_var < int(vector_varnames.size()); i_var++){
                                std::string varname = vector_varnames.at(i_var);
                                int vartype = map_vartypes[varname];
                                if (vartype == 1){
                                        int variable = variable_int_mc[varname];
                                        histogram_map_mc[varname]->Fill(variable);
                                }
                                else if (vartype == 2){
                                        double variable = variable_double_mc[varname];
                                        histogram_map_mc[varname]->Fill(variable);
                                }       
                                else if (vartype == 3){
                                        bool variable = variable_bool_mc[varname];
                                        histogram_map_mc[varname]->Fill(variable);
                                }
                                else if (vartype == 4){
                                        std::vector<double>* variable = variable_vector_mc[varname];
                                        for (int j=0; j < int(variable->size()); j++){
                                                double single_variable = variable->at(j);
                                                histogram_map_mc[varname]->Fill(single_variable);
                                        }
                                }
                        }
                }

		for (int i_var = 0; i_var < int(vector_varnames.size()); i_var++){

			std::string variable = vector_varnames.at(i_var);
			TH1F *hist_data = histogram_map_data[variable];
			TH1F *hist_mc = histogram_map_mc[variable];
			hist_data->Scale(1./hist_data->Integral());
			hist_mc->Scale(1./hist_mc->Integral());
			hist_data->SetLineColor(2);
			hist_data->SetStats(0);
			double max_single = hist_data->GetMaximum();
			double max_multi = hist_mc->GetMaximum();
			double global_max = max_single;
			if (max_multi > max_single) global_max = max_multi;

			TCanvas *c_comparison = new TCanvas(variable.c_str(),"Comparison Canvas",900,600);
			hist_data->Draw("hist");
			hist_mc->Draw("same hist");
			hist_data->GetYaxis()->SetRangeUser(0,1.1*global_max);
			leg->Clear();
			leg->AddEntry(hist_data,label1,"l");
			leg->AddEntry(hist_mc,label2,"l");
			leg->Draw("same");
			vector_canvas.push_back(c_comparison);

			if (i_var==0){
				c_comparison->Print("ClassificationVars_Comparison_data_MC_StandardLightYield_Minus20.pdf(","pdf");
			}
			else if (i_var == int(vector_varnames.size())-1){
				c_comparison->Print("ClassificationVars_Comparison_data_MC_StandardLightYield_Minus20.pdf)","pdf");
			}
			else {
				c_comparison->Print("ClassificationVars_Comparison_data_MC_StandardLightYield_Minus20.pdf","pdf");
			}
		}


		f_out->cd();
		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){
			vector_canvas.at(i_var)->Write();
		}

		f_data->cd();
		//classification_tree->Write("",TObject::kOverwrite);

		f_out->Close();
		f_data->Close();
		f_mc->Close();

		delete f_out;
		delete f_data;
		delete f_mc;

	}

}
