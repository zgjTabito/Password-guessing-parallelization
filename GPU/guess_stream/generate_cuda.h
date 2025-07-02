#pragma once
#include <vector>
#include <string>
#include "PCFG.h"

void GeneratePTGuessesBatchCUDA(const vector<PT>& pts, model& m, vector<string>& output_guesses);
