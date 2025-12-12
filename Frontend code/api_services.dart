import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Your API URL
  static const String baseUrl = "http://127.0.0.1:8000";

  // -------------------------
  // CHECK CONNECTION
  // -------------------------
  static Future<String> testConnection() async {
    final url = Uri.parse("$baseUrl/home");

    final response = await http.get(url);

    if (response.statusCode == 200) {
      return response.body;
    } else {
      throw Exception("API Error: ${response.statusCode}");
    }
  }

  // -------------------------
  // PREDICT — USE GET METHOD
  // -------------------------
  static Future<String> predictRisk({
    required String region,
    required int month,
    required int year,
  }) async {
    final url = Uri.parse(
      "$baseUrl/predict?region=$region&month=$month&year=$year",
    );

    final response = await http.get(url); // MUST BE GET

    if (response.statusCode == 200) {
      return response.body;
    } else {
      throw Exception("API Error: ${response.statusCode} → ${response.body}");
    }
  }
}