<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Services\NemaAi\NemaAiClient;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

/**
 * Routes (routes/api.php):
 *   Route::middleware(['auth:sanctum', 'throttle:20,1'])->prefix('ai')->group(function () {
 *       Route::post('listing/parse', [AiAssistController::class, 'parseListing']);
 *       Route::post('waste/sort', [AiAssistController::class, 'wasteSort']);
 *   });
 * Flutter and the web app call these routes; they never call the GPU server directly (the key stays on the server).
 */
class AiAssistController extends Controller
{
    public function __construct(private NemaAiClient $ai)
    {
    }

    public function parseListing(Request $request): JsonResponse
    {
        $data = $request->validate(['message' => 'required|string|max:1500']);
        $listing = $this->ai->extractListing($data['message']);
        if ($listing === null) {
            return response()->json(['ok' => false, 'message' => __('Could not read the message, please fill the form.')], 422);
        }
        // The model proposes; the user confirms every field in the form before the offer is published.
        return response()->json(['ok' => true, 'listing' => $listing]);
    }

    public function wasteSort(Request $request): JsonResponse
    {
        $data = $request->validate(['question' => 'required|string|max:500']);
        $answer = $this->ai->wasteSort($data['question']);
        return $answer === null
            ? response()->json(['ok' => false], 422)
            : response()->json(['ok' => true, 'answer' => $answer]);
    }
}
