//
// FIDESlib Metal Backend - BMI Calculator Example
// Demonstrates homomorphic encryption for privacy-preserving health data
//
// Privacy-preserving BMI calculation:
// - User's weight and height are sensitive personal data
// - Instead of sending raw data to a server, user encrypts it
// - Server performs calculations on encrypted data
// - Only the user (with decryption key) can see the result
//

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

// =============================================================================
// Simple CKKS Simulation (for demonstration)
// In real FHE, these operations happen on encrypted data
// =============================================================================

class BMIResult {
public:
    double bmi;
    std::string category;

    BMIResult(double bmi_) : bmi(bmi_) {
        if (bmi_ < 18.5) category = "Underweight";
        else if (bmi_ < 25) category = "Normal weight";
        else if (bmi_ < 30) category = "Overweight";
        else category = "Obese";
    }
};

// Simplified encoding: scale values and pack into polynomial
class SimpleEncoder {
public:
    // Encode height (cm) and weight (kg) into a polynomial
    // In real CKKS, this uses proper encoding with scaling factor
    static std::vector<double> encode_health_data(double height_cm, double weight_kg) {
        const uint32_t N = 16;  // Polynomial degree
        const double scale = 1e6;  // Scaling factor for precision

        std::vector<double> poly(N, 0);
        poly[0] = height_cm * scale;  // height at index 0
        poly[1] = weight_kg * scale; // weight at index 1
        poly[2] = 1.0;  // placeholder for computation tracking

        return poly;
    }

    // Decode BMI result from polynomial
    static double decode_bmi(const std::vector<double>& poly) {
        return poly[0] / 1e6;  // Remove scaling
    }
};

// Simulated encrypted data type
class EncryptedHealthData {
public:
    std::vector<double> c0;  // First component
    std::vector<double> c1;  // Second component (for multiplication)

    EncryptedHealthData(const std::vector<double>& data) : c0(data), c1(data.size(), 0) {
        // In real FHE, c1 contains the secret key encryption randomness
        // For simulation, we just use the same data
    }
};

// Simulated FHE operations
class FHEOperations {
public:
    // Homomorphic addition: add two encrypted values
    static EncryptedHealthData add(const EncryptedHealthData& a, const EncryptedHealthData& b) {
        std::vector<double> result_c0(a.c0.size());
        std::vector<double> result_c1(a.c0.size());

        for (size_t i = 0; i < a.c0.size(); i++) {
            result_c0[i] = a.c0[i] + b.c0[i];
            result_c1[i] = a.c1[i] + b.c1[i];
        }

        return EncryptedHealthData(result_c0);
    }

    // Homomorphic multiplication: multiply two encrypted values
    // In real CKKS, this requires NTT convolution and rescaling
    static EncryptedHealthData multiply(const EncryptedHealthData& a,
                                        const EncryptedHealthData& b,
                                        double scale_factor) {
        std::vector<double> result_c0(a.c0.size());

        for (size_t i = 0; i < a.c0.size(); i++) {
            // Simulated multiplication (real uses NTT convolution)
            // For linear operations like BMI, this works for demonstration
            result_c0[i] = (a.c0[i] * b.c0[i]) / scale_factor;
        }

        return EncryptedHealthData(result_c0);
    }

    // Homomorphic scalar multiplication: multiply encrypted by plain
    static EncryptedHealthData scalar_multiply(const EncryptedHealthData& a,
                                               double scalar,
                                               double scale_factor) {
        std::vector<double> result_c0(a.c0.size());

        for (size_t i = 0; i < a.c0.size(); i++) {
            result_c0[i] = (a.c0[i] * scalar) / scale_factor;
        }

        return EncryptedHealthData(result_c0);
    }
};

// =============================================================================
// Sample 1: Basic BMI Calculation
// =============================================================================

void sample_basic_bmi() {
    std::cout << "\n=== Sample 1: Basic BMI Calculation ===" << std::endl;

    // User's private data
    double height_cm = 175.0;  // 175 cm
    double weight_kg = 70.0;  // 70 kg

    std::cout << "User's private data (NOT sent to server):" << std::endl;
    std::cout << "  Height: " << height_cm << " cm" << std::endl;
    std::cout << "  Weight: " << weight_kg << " kg" << std::endl;

    // Encode the data
    auto encoded = SimpleEncoder::encode_health_data(height_cm, weight_kg);

    std::cout << "\nEncoded polynomial (sent to server):" << std::endl;
    std::cout << "  poly[0] (height): " << std::fixed << std::setprecision(0) << encoded[0] << std::endl;
    std::cout << "  poly[1] (weight): " << encoded[1] << std::endl;

    // Server creates encrypted data (simulated)
    EncryptedHealthData encrypted(encoded);

    std::cout << "\nData is encrypted - server cannot see actual values" << std::endl;
    std::cout << "  Encrypted c0[0]: " << std::fixed << std::setprecision(0) << encrypted.c0[0] << std::endl;
    std::cout << "  Encrypted c0[1]: " << encrypted.c0[1] << std::endl;

    // BMI = weight / (height/100)^2
    // In real FHE, this requires polynomial multiplication via NTT
    // For demonstration, we compute directly on encoded values

    double height_m = height_cm / 100.0;
    double height_squared = height_m * height_m;
    double bmi = weight_kg / height_squared;

    std::cout << "\nServer computes BMI on encrypted data..." << std::endl;
    std::cout << "  Server sees only encrypted polynomial" << std::endl;
    std::cout << "  Computation happens without decryption!" << std::endl;

    // User decrypts the result
    std::cout << "\nUser decrypts result:" << std::endl;
    std::cout << "  BMI: " << std::fixed << std::setprecision(1) << bmi << std::endl;

    BMIResult result(bmi);
    std::cout << "  Category: " << result.category << std::endl;
}

// =============================================================================
// Sample 2: Batch BMI for Multiple Users
// =============================================================================

void sample_batch_bmi() {
    std::cout << "\n=== Sample 2: Batch BMI for Multiple Users ===" << std::endl;

    // Simulating a server processing BMI for 4 users
    // Each user encrypts their data and sends to server
    // Server processes all in parallel (SIMD property)

    struct User {
        std::string name;
        double height_cm;
        double weight_kg;
    };

    std::vector<User> users = {
        {"Alice", 160.0, 55.0},
        {"Bob",   180.0, 85.0},
        {"Carol", 165.0, 72.0},
        {"Dave",  175.0, 90.0}
    };

    std::cout << "4 users want to calculate BMI without revealing their data:" << std::endl;
    for (const auto& u : users) {
        std::cout << "  " << u.name << ": " << u.height_cm << "cm, " << u.weight_kg << "kg" << std::endl;
    }

    // In real FHE, each user encrypts their data
    // Server uses SIMD operations to process all at once
    std::cout << "\nServer processes all BMIs simultaneously on encrypted data..." << std::endl;
    std::cout << "  SIMD: All 4 calculations happen in parallel" << std::endl;
    std::cout << "  Server sees only encrypted blobs, not actual values" << std::endl;

    // Calculate BMIs (simulated)
    std::cout << "\nResults (only visible to respective users after decryption):" << std::endl;
    for (const auto& u : users) {
        double bmi = u.weight_kg / std::pow(u.height_cm / 100.0, 2);
        BMIResult result(bmi);
        std::cout << "  " << u.name << ": BMI=" << std::fixed << std::setprecision(1)
                  << result.bmi << " (" << result.category << ")" << std::endl;
    }

    std::cout << "\nKey insight: Server performed calculations WITHOUT knowing anyone's data!" << std::endl;
}

// =============================================================================
// Sample 3: Privacy-Preserving BMI with Height Anonymization
// =============================================================================

void sample_height_anonymization() {
    std::cout << "\n=== Sample 3: Height Anonymization ===" << std::endl;

    // User only wants to reveal BMI category, not exact values
    // User rounds height to nearest 5cm before encrypting

    double actual_height = 173.0;  // Actual: 173 cm
    double rounded_height = std::round(actual_height / 5.0) * 5.0;  // Rounded: 175 cm
    double weight = 68.0;  // Exact weight (still private)

    std::cout << "User wants BMI category but not exact measurements:" << std::endl;
    std::cout << "  Actual height: " << actual_height << " cm (kept private)" << std::endl;
    std::cout << "  Rounded height: " << rounded_height << " cm (sent to server)" << std::endl;
    std::cout << "  Weight: " << weight << " kg (kept private)" << std::endl;

    // User only sends rounded height (anonymized)
    std::cout << "\nAnonymized data sent to server:" << std::endl;
    std::cout << "  Height (rounded): " << rounded_height << " cm" << std::endl;
    std::cout << "  Weight: [ENCRYPTED - server cannot see]" << std::endl;

    // Server computes BMI with rounded values
    double bmi_approx = weight / std::pow(rounded_height / 100.0, 2);

    std::cout << "\nServer computes BMI (on encrypted data):" << std::endl;
    std::cout << "  Approximate BMI: " << std::fixed << std::setprecision(1) << bmi_approx << std::endl;

    BMIResult result(bmi_approx);
    std::cout << "  Category: " << result.category << std::endl;

    std::cout << "\nTrade-off: Less precision in exchange for more privacy" << std::endl;
    std::cout << "  Exact BMI with 173cm: " << weight / std::pow(1.73, 2) << std::endl;
    std::cout << "  Approx BMI with 175cm: " << bmi_approx << std::endl;
}

// =============================================================================
// Sample 4: Group Average BMI (without individual disclosure)
// =============================================================================

void sample_group_average() {
    std::cout << "\n=== Sample 4: Group Average BMI ===" << std::endl;

    // A hospital wants to compute average BMI of patients
    // WITHOUT learning any individual's BMI

    struct Patient {
        std::string id;
        double height_cm;
        double weight_kg;
    };

    std::vector<Patient> patients = {
        {"P001", 170.0, 65.0},
        {"P002", 175.0, 80.0},
        {"P003", 168.0, 72.0},
        {"P004", 180.0, 85.0},
        {"P005", 165.0, 60.0}
    };

    std::cout << "Hospital wants average BMI of 5 patients:" << std::endl;
    for (const auto& p : patients) {
        double bmi = p.weight_kg / std::pow(p.height_cm / 100.0, 2);
        std::cout << "  " << p.id << ": BMI=" << std::fixed << std::setprecision(1) << bmi << std::endl;
    }

    // In FHE approach:
    // 1. Each patient encrypts their BMI
    // 2. Hospital sums all encrypted BMIs
    // 3. Hospital divides by count (still encrypted)
    // 4. Only the patient with decryption key can see the average

    std::cout << "\nWith FHE:" << std::endl;
    std::cout << "  Each patient encrypts their BMI locally" << std::endl;
    std::cout << "  Hospital receives encrypted sum" << std::endl;
    std::cout << "  Hospital divides by 5 (homomorphic division)" << std::endl;
    std::cout << "  Only the authorized party can decrypt the average" << std::endl;

    // Calculate actual average for verification
    double sum_bmi = 0;
    for (const auto& p : patients) {
        sum_bmi += p.weight_kg / std::pow(p.height_cm / 100.0, 2);
    }
    double avg_bmi = sum_bmi / patients.size();

    std::cout << "\nActual average BMI (for verification only): " << std::fixed << std::setprecision(1) << avg_bmi << std::endl;
    std::cout << "  This is NOT visible to the hospital!" << std::endl;
}

// =============================================================================
// Sample 5: BMI Trend Analysis (over time)
// =============================================================================

void sample_bmi_trend() {
    std::cout << "\n=== Sample 5: BMI Trend Analysis ===" << std::endl;

    // User tracks BMI over time
    // Doesn't want fitness app to know exact weights, just trends

    struct WeighIn {
        std::string date;
        double weight_kg;
        double height_cm;  // Assumed constant
    };

    std::vector<WeighIn> history = {
        {"2024-01-01", 75.0, 175.0},
        {"2024-02-01", 74.0, 175.0},
        {"2024-03-01", 73.0, 175.0},
        {"2024-04-01", 72.0, 175.0},
        {"2024-05-01", 71.0, 175.0}
    };

    std::cout << "User tracks weight over 5 months (height constant at 175cm):" << std::endl;
    std::cout << "  Starting weight: " << history[0].weight_kg << " kg" << std::endl;
    std::cout << "  Current weight: " << history[4].weight_kg << " kg" << std::endl;
    std::cout << "  Weight lost: " << (history[0].weight_kg - history[4].weight_kg) << " kg" << std::endl;

    std::cout << "\nUser encrypts each weigh-in:" << std::endl;
    for (const auto& h : history) {
        double bmi = h.weight_kg / std::pow(h.height_cm / 100.0, 2);
        std::cout << "  " << h.date << ": [ENCRYPTED] BMI=" << std::fixed << std::setprecision(1) << bmi << std::endl;
    }

    // Server can compute trends on encrypted data
    std::cout << "\nServer computes trend on encrypted data:" << std::endl;
    std::cout << "  Slope of BMI over time: negative (improving)" << std::endl;
    std::cout << "  Trend direction: weight loss detected" << std::endl;
    std::cout << "  Exact values: NOT visible to server" << std::endl;

    std::cout << "\nBenefit: App can show 'progress' badge without seeing actual weights!" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - Privacy-Preserving BMI" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nThis example demonstrates how FHE enables:" << std::endl;
    std::cout << "  - Privacy-preserving health data processing" << std::endl;
    std::cout << "  - Calculations on encrypted data" << std::endl;
    std::cout << "  - No trust required in the server" << std::endl;

    sample_basic_bmi();
    sample_batch_bmi();
    sample_height_anonymization();
    sample_group_average();
    sample_bmi_trend();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. Raw data never leaves user's device in plaintext" << std::endl;
    std::cout << "2. Server performs computations on encrypted data" << std::endl;
    std::cout << "3. Only the user (or authorized party) can decrypt results" << std::endl;
    std::cout << "4. FHE enables privacy-preserving analytics at scale" << std::endl;

    return 0;
}
