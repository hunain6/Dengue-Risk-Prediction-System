import 'dart:convert';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class PredictionInputScreen extends StatefulWidget {
  const PredictionInputScreen({super.key});

  @override
  State<PredictionInputScreen> createState() => _PredictionInputScreenState();
}

class _PredictionInputScreenState extends State<PredictionInputScreen> {
  final List<String> regions = [
    "Punjab",
    "Sindh",
    "KP",
    "Balochistan",
    "AJK",
    "ICT",
    "GB",
  ];

  final List<String> months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  String? selectedRegion;
  String? selectedMonth;
  String year = "";

  Map<String, dynamic>? result;

  bool loading = false;

  void predictNow() async {
    if (selectedRegion == null || selectedMonth == null || year.isEmpty) return;

    setState(() => loading = true);

    try {
      String response = await ApiService.predictRisk(
        region: selectedRegion!,
        month: months.indexOf(selectedMonth!) + 1,
        year: int.parse(year),
      );

      final decoded = json.decode(response);

      setState(() {
        result = {
          "risk_category": decoded["risk_category"]?.toString() ?? "N/A",
          "risk_score": decoded["risk_score"]?.toString() ?? "0",
          "values_used": {
            "dengue_cases":
                decoded["values_used"]["dengue_cases"]?.toString() ?? "0",
            "dengue_deaths":
                decoded["values_used"]["dengue_deaths"]?.toString() ?? "0",
            "temperature":
                decoded["values_used"]["temperature"]?.toString() ?? "0",
            "humidity": decoded["values_used"]["humidity"]?.toString() ?? "0",
            "rainfall": decoded["values_used"]["rainfall"]?.toString() ?? "0",
          },
        };
      });
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Prediction Error: $e")));
    }

    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Dengue Risk Prediction"),
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            // LEFT SIDE INPUTS
            Expanded(
              flex: 4,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Enter Details",
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 20),

                  // REGION
                  DropdownButtonFormField(
                    decoration: const InputDecoration(
                      labelText: "Select Region",
                      border: OutlineInputBorder(),
                    ),
                    value: selectedRegion,
                    items: regions
                        .map((r) => DropdownMenuItem(value: r, child: Text(r)))
                        .toList(),
                    onChanged: (v) => setState(() => selectedRegion = v),
                  ),
                  const SizedBox(height: 20),

                  // MONTH
                  DropdownButtonFormField(
                    decoration: const InputDecoration(
                      labelText: "Select Month",
                      border: OutlineInputBorder(),
                    ),
                    value: selectedMonth,
                    items: months
                        .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                        .toList(),
                    onChanged: (v) => setState(() => selectedMonth = v),
                  ),
                  const SizedBox(height: 20),

                  // YEAR
                  TextField(
                    decoration: const InputDecoration(
                      labelText: "Enter Year",
                      border: OutlineInputBorder(),
                    ),
                    keyboardType: TextInputType.number,
                    onChanged: (v) => year = v,
                  ),
                  const SizedBox(height: 25),

                  // PREDICT BUTTON
                  ElevatedButton(
                    onPressed: loading ? null : predictNow,
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 55),
                      backgroundColor: Colors.teal,
                    ),
                    child: loading
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Text("PREDICT", style: TextStyle(fontSize: 18)),
                  ),
                ],
              ),
            ),

            const SizedBox(width: 30),

            // RIGHT SIDE RESULT CARD
            Expanded(
              flex: 6,
              child: result == null
                  ? const Center(
                      child: Text(
                        "Prediction Result Will Appear Here",
                        style: TextStyle(fontSize: 18, color: Colors.grey),
                      ),
                    )
                  : Card(
                      elevation: 10,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(25),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            // BIG RISK CIRCLE
                            CircleAvatar(
                              radius: 60,
                              backgroundColor: Colors.teal.shade700,
                              child: Text(
                                result!["risk_category"],
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            const SizedBox(height: 20),

                            Text(
                              "Risk Score: ${result!["risk_score"]}",
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 20),
                            const Divider(),

                            _valueRow(
                              "Total Cases",
                              result!["values_used"]["dengue_cases"],
                            ),
                            _valueRow(
                              "Total Deaths",
                              result!["values_used"]["dengue_deaths"],
                            ),
                            _valueRow(
                              "Temperature",
                              result!["values_used"]["temperature"],
                            ),
                            _valueRow(
                              "Humidity",
                              result!["values_used"]["humidity"],
                            ),
                            _valueRow(
                              "Rainfall",
                              result!["values_used"]["rainfall"],
                            ),
                          ],
                        ),
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _valueRow(String title, dynamic value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
          ),
          Text(
            value.toString(),
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}