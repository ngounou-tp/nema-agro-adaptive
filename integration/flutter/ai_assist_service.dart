// Flutter side: send a pasted WhatsApp message to the Laravel API and pre-fill the "new offer" form.
// Add to pubspec.yaml:  http: ^1.2.0
import 'dart:convert';
import 'package:http/http.dart' as http;

class AiAssistService {
  AiAssistService({required this.baseUrl, required this.token});
  final String baseUrl; // e.g. https://api.nema.cm/api
  final String token;   // the user's Sanctum token

  /// Returns the parsed listing, or null when the model could not read the message
  /// (then show the empty form as usual).
  Future<Map<String, dynamic>?> parseListing(String message) async {
    final res = await http
        .post(
          Uri.parse('$baseUrl/ai/listing/parse'),
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: jsonEncode({'message': message}),
        )
        .timeout(const Duration(seconds: 70));
    if (res.statusCode != 200) return null;
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    return body['ok'] == true ? body['listing'] as Map<String, dynamic> : null;
  }
}

// Usage in the offer screen:
//   final listing = await AiAssistService(baseUrl: apiUrl, token: userToken).parseListing(pastedText);
//   if (listing != null) {
//     productCtrl.text = listing['product'] ?? '';
//     quantityCtrl.text = '${listing['quantity'] ?? ''}';
//     priceCtrl.text = '${listing['price'] ?? ''}';
//     // ... then let the user check and confirm every field before publishing.
//   }
