<?php

namespace App\Services\NemaAi;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use RuntimeException;

/**
 * Calls the NEMA-Agro / REVALO-Waste adapters served by vLLM (OpenAI-compatible API).
 *
 * Install:
 *  1. Copy this file to app/Services/NemaAi/NemaAiClient.php
 *  2. Copy demo/tasks_agro.json and demo/tasks_waste.json to storage/app/nema_ai/
 *  3. Add to config/services.php:
 *       'nema_ai' => [
 *           'url' => env('NEMA_AI_URL'),          // e.g. https://your-gpu-host:8000/v1
 *           'key' => env('NEMA_AI_KEY'),          // the VLLM_API_KEY you chose
 *           'timeout' => env('NEMA_AI_TIMEOUT', 60),
 *       ],
 *  4. Add NEMA_AI_URL and NEMA_AI_KEY to .env
 *
 * The prompt MUST be built exactly as in training: optional "Context:" block, then the task instruction,
 * then the input. Never change the instruction text without retraining.
 */
class NemaAiClient
{
    /** Keys each structured task must return (the client rejects anything else). */
    private const EXPECTED_KEYS = [
        'agro_listing_extract' => ['intent', 'product', 'quantity', 'unit', 'quantity_kg', 'price', 'price_basis', 'currency', 'location', 'available_from', 'quality', 'contact'],
        'agro_unit_price' => ['quantity_kg', 'price_per_kg', 'total_price', 'reference_price_per_kg', 'difference_pct'],
        'agro_coop_ledger' => ['entries', 'totals_kg'],
        'agro_trace_record' => ['lot_id', 'producer', 'group', 'product', 'harvest_date', 'quantity_kg', 'last_treatment', 'last_treatment_date', 'phi_days', 'phi_respected'],
        'agro_symptom_triage' => ['crop', 'problem', 'see_technician', 'advice'],
        'agro_input_safety' => ['decision', 'dose', 'message'],
        'waste_sort' => ['category', 'valorisable', 'hazardous', 'route'],
        'waste_pickup_extract' => ['city', 'neighbourhood', 'waste_type', 'bags', 'urgency', 'contact'],
        'waste_compost_calc' => ['input_kg', 'organic_kg', 'compost_kg', 'value_xaf'],
    ];

    private array $tasks = [];

    public function __construct()
    {
        foreach (['agro', 'waste'] as $p) {
            $path = storage_path("app/nema_ai/tasks_{$p}.json");
            if (is_file($path)) {
                $this->tasks += json_decode(file_get_contents($path), true);
            }
        }
    }

    /** Marketplace message -> listing array (to pre-fill the "new offer" form). */
    public function extractListing(string $message, ?string $messageDate = null): ?array
    {
        $input = 'Message date: ' . ($messageDate ?? now()->toDateString()) . "\nMessage: " . $message;
        return $this->run('nema-agro', 'agro_listing_extract', $input);
    }

    /** Symptom description -> triage. Pass the problem table rows for the crop as context (from your DB). */
    public function triage(string $description, string $problemTable): ?array
    {
        return $this->run('nema-agro', 'agro_symptom_triage', 'Farmer: ' . $description, $problemTable);
    }

    /** Pesticide question -> safe answer. Context = registered products for the crop (from the official list in your DB). */
    public function inputSafety(string $question, string $registeredProducts): ?array
    {
        return $this->run('nema-agro', 'agro_input_safety', 'Question: ' . $question, $registeredProducts);
    }

    public function wasteSort(string $question): ?array
    {
        return $this->run('revalo-waste', 'waste_sort', 'Question: ' . $question, '');
    }

    /**
     * Generic call. $context = null uses the default context saved with the task (e.g. product and unit codes).
     * Returns the decoded JSON, or null if the model failed twice (log + let the user fill the form by hand).
     */
    public function run(string $model, string $task, string $input, ?string $context = null): ?array
    {
        $t = $this->tasks[$task] ?? throw new RuntimeException("Unknown task {$task}: copy tasks_*.json to storage/app/nema_ai/");
        $context = $context ?? ($t['context'] ?? '');
        $prompt = $t['instruction'] . "\n" . $input;
        if (trim($context) !== '') {
            $prompt = "Context:\n" . trim($context) . "\n\n" . $prompt;
        }

        for ($attempt = 1; $attempt <= 2; $attempt++) {
            $response = Http::withToken(config('services.nema_ai.key'))
                ->timeout((int) config('services.nema_ai.timeout', 60))
                ->post(rtrim(config('services.nema_ai.url'), '/') . '/chat/completions', [
                    'model' => $model,
                    'temperature' => 0,
                    'max_tokens' => $task === 'agro_coop_ledger' ? 900 : 300,
                    'messages' => [['role' => 'user', 'content' => $prompt]],
                ]);

            if ($response->failed()) {
                Log::warning('nema_ai http error', ['task' => $task, 'status' => $response->status()]);
                continue;
            }
            $text = (string) data_get($response->json(), 'choices.0.message.content', '');
            $json = $this->extractJson($text);
            if ($json !== null && $this->hasKeys($task, $json)) {
                return $json;
            }
            Log::info('nema_ai invalid output', ['task' => $task, 'attempt' => $attempt, 'output' => mb_substr($text, 0, 500)]);
        }
        return null;
    }

    private function extractJson(string $text): ?array
    {
        $text = preg_replace('/<think>.*?<\/think>/s', '', $text);
        $text = trim(preg_replace('/^```(?:json)?|```$/m', '', $text));
        $start = strpos($text, '{');
        $end = strrpos($text, '}');
        if ($start === false || $end === false) {
            return null;
        }
        $decoded = json_decode(substr($text, $start, $end - $start + 1), true);
        return is_array($decoded) ? $decoded : null;
    }

    private function hasKeys(string $task, array $json): bool
    {
        $want = self::EXPECTED_KEYS[$task] ?? null;
        if ($want === null) {
            return true;
        }
        $have = array_keys($json);
        sort($want);
        sort($have);
        return $want === $have;
    }
}
