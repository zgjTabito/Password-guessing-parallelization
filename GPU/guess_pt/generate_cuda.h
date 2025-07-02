#pragma once
#include <vector>
#include <string>
#include "PCFG.h"

void GeneratePTGuessesBatchCUDA(
    const std::vector<PT>& pts,
    model& m,
    std::vector<std::string>& output_guesses
);
