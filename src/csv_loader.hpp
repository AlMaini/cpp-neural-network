#ifndef CSV_LOADER
#define CSV_LOADER

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Structure to hold a single MNIST record
struct Record {
    int label;
    std::vector<int> pixels;
};

// Class to load and manage CSV data
class CSVLoader {
public:
    CSVLoader(const std::string& filename) : filename_(filename) {}
    
    bool load() {
        std::ifstream file(filename_);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename_ << std::endl;
            return false;
        }
        
        std::string line;
        // Skip header line
        std::getline(file, line);
        
        // Read data lines
        while (std::getline(file, line)) {
            Record record;
            std::stringstream ss(line);
            std::string value;
            
            // First value is the label
            if (std::getline(ss, value, ',')) {
                record.label = std::stoi(value);
            }
            
            // Remaining values are pixels
            while (std::getline(ss, value, ',')) {
                record.pixels.push_back(std::stoi(value));
            }
            
            data_.push_back(record);
        }
        
        file.close();
        return true;
    }
    
    // Get all records
    const std::vector<Record>& getData() const {
        return data_;
    }
    
    // Get specific record
    const Record& getRecord(size_t index) const {
        return data_[index];
    }
    
    // Get number of records
    size_t size() const {
        return data_.size();
    }
    
    // Print statistics
    void printStats() const {
        if (data_.empty()) {
            std::cout << "No data loaded" << std::endl;
            return;
        }
        
        std::cout << "Total records: " << data_.size() << std::endl;
        std::cout << "Pixels per image: " << data_[0].pixels.size() << std::endl;
        
        // Count labels
        std::vector<int> labelCounts(10, 0);
        for (const auto& record : data_) {
            if (record.label >= 0 && record.label < 10) {
                labelCounts[record.label]++;
            }
        }
        
        std::cout << "Label distribution:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "  " << i << ": " << labelCounts[i] << std::endl;
        }
    }
    
    // Print a single image in ASCII art
    void printImage(size_t index) const {
        if (index >= data_.size()) {
            std::cerr << "Error: Index out of bounds" << std::endl;
            return;
        }
        
        const auto& record = data_[index];
        std::cout << "Label: " << record.label << std::endl;
        
        // Assuming 28x28 image
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixel = record.pixels[i * 28 + j];
                if (pixel > 127) {
                    std::cout << "# ";
                } else if (pixel > 64) {
                    std::cout << ". ";
                } else {
                    std::cout << "  ";
                }
            }
            std::cout << std::endl;
        }
    }

private:
    std::string filename_;
    std::vector<Record> data_;
};

#endif // CSV_LOADER_H